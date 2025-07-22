import numpy as np
from PIL import Image, ImageDraw, ImageStat
import torch
from comfy.utils import ProgressBar
import random
import math
import cv2


class PixelScatterNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PixelScatterNode (Point Scatter Glitch Style) using device: {self.device}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "base_block_size": ("INT", { "default": 4, "min": 1, "max": 32, "step": 1 }), # Size of analysis block
                "render_threshold": ("FLOAT", { # Luminance below this won't render/scatter
                     "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "stochastic_drop_chance": ("FLOAT", { # Chance (0-1) to randomly skip rendering a bright block
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "max_jitter_distance": ("INT", { # Max random displacement distance in pixels
                    "default": 100, "min": 0, "max": 1024, "step": 1 }), # Increased max
                "jitter_luminance_power": ("FLOAT", { # Controls how block brightness scales jitter range (>1 = brighter jitters further)
                     "default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1 }),
                # --- Appearance ---
                "color_mode": (["Original Color", "Matrix Color"], {"default": "Original Color"}),
                "matrix_color": ("COLOR", {"default": "#20FF80"}),
                "matrix_intensity_scale": ("FLOAT", { "default": 1.8, "min": 0.1, "max": 10.0, "step": 0.05 }), # Increased range
                "size_mode": (["Fixed", "Variable"], {"default": "Variable"}),
                "min_draw_size": ("INT", {"default": 1, "min": 1, "max": 32, "step": 1}), # Min size if variable
                "size_luminance_power": ("FLOAT", { # How strongly luminance affects variable size
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "alpha_luminance_power": ("FLOAT", { # How strongly luminance affects brightness (alpha sim)
                    "default": 0.8, "min": 0.0, "max": 3.0, "step": 0.05}), # 0 disables
                # --- End Appearance ---
                "background_color": ("COLOR", {"default": "#000000"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "mask": ("MASK",), # Mask applies to where the effect is calculated
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "SyntaxNodes/VisualEffects" # Or Glitch

    def get_mask_array(self, mask, target_shape):
        """Convert mask to proper format and shape - copied from working node"""
        if mask is None:
            return np.ones((target_shape[0], target_shape[1]), dtype=np.float32)
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(mask):
            # Handle different tensor formats
            if len(mask.shape) == 4:  # BCHW format
                mask = mask[0, 0]  # Take first batch, first channel
            elif len(mask.shape) == 3:  # CHW format
                mask = mask[0]  # Take first channel
            mask = mask.cpu().numpy()
        
        # Ensure mask is float32 and properly scaled
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        # Resize mask to match target shape
        if mask.shape[0] != target_shape[0] or mask.shape[1] != target_shape[1]:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), 
                            interpolation=cv2.INTER_LINEAR)
        
        return mask

    def process_image(self, image, base_block_size, render_threshold, stochastic_drop_chance,
                      max_jitter_distance, jitter_luminance_power,
                      color_mode, matrix_color, matrix_intensity_scale,
                      size_mode, min_draw_size, size_luminance_power, alpha_luminance_power,
                      background_color, seed, mask=None):
        # --- Setup ---
        device = image.device
        batch_size, H, W, _ = image.shape
        pbar = ProgressBar(batch_size)
        bg_color_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        matrix_color_rgb = tuple(int(matrix_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        processed_tensors = []
        master_seed = seed

        for idx in range(batch_size):
            current_seed = master_seed + idx
            random.seed(current_seed)
            np.random.seed(current_seed)

            single_image_tensor = image[idx:idx+1]

            pil_image = self.t2p(single_image_tensor)
            if pil_image is None:
                print(f"Warn: Skipping image {idx}: conversion error.")
                placeholder = Image.new('RGB', (W, H), bg_color_rgb)
                processed_tensors.append(self.p2t(placeholder)); continue

            pil_image_rgb = pil_image.convert('RGB')
            
            # Get mask for this frame using the robust method
            mask_array = self.get_mask_array(
                mask[idx] if mask is not None and idx < mask.shape[0] else None, 
                (H, W)
            )

            # --- Create base and effect images ---
            if mask is not None:
                # Start with original image as base
                base_image = pil_image_rgb.copy()
                # Create effect image with scatter effect
                effect_image = self.create_point_scatter_with_mask(
                    pil_image_rgb, mask_array, base_block_size, bg_color_rgb,
                    render_threshold, stochastic_drop_chance,
                    max_jitter_distance, jitter_luminance_power,
                    color_mode, matrix_color_rgb, matrix_intensity_scale,
                    size_mode, min_draw_size, size_luminance_power, alpha_luminance_power
                )
                
                # Blend base and effect using mask
                base_array = np.array(base_image)
                effect_array = np.array(effect_image)
                mask_array_3d = np.stack([mask_array] * 3, axis=-1)
                
                final_array = base_array * (1 - mask_array_3d) + effect_array * mask_array_3d
                processed_image = Image.fromarray(final_array.astype(np.uint8))
            else:
                # No mask - apply to entire image as before
                processed_image = self.create_point_scatter(
                    pil_image_rgb, base_block_size, bg_color_rgb,
                    render_threshold, stochastic_drop_chance,
                    max_jitter_distance, jitter_luminance_power,
                    color_mode, matrix_color_rgb, matrix_intensity_scale,
                    size_mode, min_draw_size, size_luminance_power, alpha_luminance_power
                )

            # --- Output ---
            processed_tensor = self.p2t(processed_image)
            if processed_tensor is not None: 
                processed_tensors.append(processed_tensor)
            pbar.update_absolute(idx + 1)

        # --- Batch Concat ---
        if not processed_tensors: 
            print("Error: No images processed."); return (image,)
        tensors_on_device = [t.to(device) for t in processed_tensors]
        try: 
            final_output = torch.cat(tensors_on_device, dim=0)
        except Exception as e: 
            print(f"Error concatenating: {e}"); return(image,)
        return (final_output,)

    def create_point_scatter_with_mask(self, image_rgb, mask_array, block_size, background_color,
                                     render_thresh, drop_chance, max_jitter, jitter_power,
                                     color_mode, matrix_rgb, matrix_intensity,
                                     size_mode, min_size, size_power, alpha_power):
        """Create scatter effect - mask is applied per-pixel during effect calculation"""
        width, height = image_rgb.size
        if width == 0 or height == 0: 
            return Image.new('RGB', (1, 1), background_color)

        W_blocks = math.ceil(width / block_size)
        H_blocks = math.ceil(height / block_size)

        # Create output image with background
        output_img = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(output_img)

        for yb in range(H_blocks):
            for xb in range(W_blocks):
                x_start, y_start = xb * block_size, yb * block_size
                x_end, y_end = min(width, x_start + block_size), min(height, y_start + block_size)
                
                # Get average mask value for this block
                block_mask = mask_array[y_start:y_end, x_start:x_end]
                mask_value = np.mean(block_mask) if block_mask.size > 0 else 0.0
                
                # Skip if mask value is too low
                if mask_value < 0.01:
                    continue

                # Get color and luminance for this block
                sample_box = (x_start, y_start, x_end, y_end)
                region = image_rgb.crop(sample_box)
                if region.size[0] <= 0 or region.size[1] <= 0:
                    continue
                
                try:
                    stat = ImageStat.Stat(region)
                    avg_color_float = stat.mean[:3]
                    avg_color_rgb = tuple(int(c) for c in avg_color_float)
                    luminance = self.calculate_luminance(avg_color_rgb)
                except Exception:
                    continue

                # Apply mask to luminance - this is the key!
                effective_luminance = luminance * mask_value

                # Apply effect logic
                if effective_luminance >= render_thresh and random.random() >= drop_chance:
                    # Calculate luminance scale (0-1 above threshold)
                    lumi_scale = max(0.0, min(1.0, (effective_luminance - render_thresh) / (1.0 - render_thresh + 1e-6)))

                    # Calculate jitter
                    current_max_jitter = max_jitter * (lumi_scale ** jitter_power)
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(0, current_max_jitter)
                    dx = distance * math.cos(angle)
                    dy = distance * math.sin(angle)

                    # Calculate draw position
                    orig_center_x = (xb + 0.5) * block_size
                    orig_center_y = (yb + 0.5) * block_size
                    draw_center_x = orig_center_x + dx
                    draw_center_y = orig_center_y + dy

                    # Determine appearance
                    base_color = avg_color_rgb
                    draw_color = base_color
                    if color_mode == "Matrix Color":
                        intensity = max(0.0, min(1.0, effective_luminance)) * matrix_intensity
                        draw_color = self.scale_color_brightness(matrix_rgb, intensity)

                    # Apply alpha simulation
                    if alpha_power > 0:
                        alpha_factor = lumi_scale ** alpha_power
                        draw_color = self.scale_color_brightness(draw_color, alpha_factor)

                    # Determine size
                    draw_size = block_size
                    if size_mode == "Variable":
                        size_factor = lumi_scale ** size_power
                        draw_size = max(min_size, int(block_size * size_factor))

                    # Draw the scattered point
                    if draw_size >= 1:
                        half_size = draw_size / 2.0
                        x0 = int(draw_center_x - half_size)
                        y0 = int(draw_center_y - half_size)
                        x1 = int(draw_center_x + half_size)
                        y1 = int(draw_center_y + half_size)
                        draw.rectangle([x0, y0, x1, y1], fill=draw_color)

        return output_img

    # --- Helpers ---
    def calculate_luminance(self, rgb_tuple):
        if not isinstance(rgb_tuple, tuple) or len(rgb_tuple) < 3: return 0.0
        r, g, b = [x / 255.0 for x in rgb_tuple[:3]]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def scale_color_brightness(self, color_rgb, scale_factor):
        """Multiplies RGB values by factor, clamps 0-255."""
        scale_factor = max(0.0, scale_factor) # Ensure non-negative
        r = max(0, min(255, int(color_rgb[0] * scale_factor)))
        g = max(0, min(255, int(color_rgb[1] * scale_factor)))
        b = max(0, min(255, int(color_rgb[2] * scale_factor)))
        return (r, g, b)

    # --- Original full-image effect (kept for when no mask) ---
    def create_point_scatter(self, image_rgb, block_size, background_color,
                             render_thresh, drop_chance, max_jitter, jitter_power,
                             color_mode, matrix_rgb, matrix_intensity,
                             size_mode, min_size, size_power, alpha_power):
        """Scatters points/blocks based on luminance and randomness."""
        width, height = image_rgb.size
        if width == 0 or height == 0: return Image.new('RGB', (1, 1), background_color)

        W_blocks = math.ceil(width / block_size)
        H_blocks = math.ceil(height / block_size)

        # 1. Analyze all blocks: Store color and luminance
        block_data = {} # Store {(xb, yb): {'color': (r,g,b), 'lumi': 0-1}}
        for yb in range(H_blocks):
            for xb in range(W_blocks):
                x_start, y_start = xb * block_size, yb * block_size
                x_end, y_end = min(width, x_start + block_size), min(height, y_start + block_size)
                sample_box = (x_start, y_start, x_end, y_end)
                region = image_rgb.crop(sample_box)
                if region.size[0] > 0 and region.size[1] > 0:
                    try:
                        stat = ImageStat.Stat(region); avg_color_float = stat.mean[:3]
                        avg_color_rgb = tuple(int(c) for c in avg_color_float)
                        luminance = self.calculate_luminance(avg_color_rgb)
                        block_data[(xb, yb)] = {'color': avg_color_rgb, 'lumi': luminance}
                    except Exception: block_data[(xb, yb)] = {'color': background_color, 'lumi': 0.0}
                else: block_data[(xb, yb)] = {'color': background_color, 'lumi': 0.0}

        # 2. Create output image and draw scattered points/blocks
        output_img = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(output_img)

        for yb in range(H_blocks):
            for xb in range(W_blocks):
                data = block_data.get((xb, yb))
                if not data: continue

                lumi = data['lumi']

                # Check threshold and random drop chance
                if lumi >= render_thresh and random.random() >= drop_chance:
                    # Calculate luminance scale (0-1 above threshold)
                    lumi_scale = max(0.0, min(1.0, (lumi - render_thresh) / (1.0 - render_thresh + 1e-6)))

                    # Calculate max jitter distance for this block
                    current_max_jitter = max_jitter * (lumi_scale ** jitter_power)

                    # Calculate random offset
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(0, current_max_jitter)
                    dx = distance * math.cos(angle)
                    dy = distance * math.sin(angle)

                    # Calculate target draw coordinates (center of block + offset)
                    orig_center_x = (xb + 0.5) * block_size
                    orig_center_y = (yb + 0.5) * block_size
                    draw_center_x = orig_center_x + dx
                    draw_center_y = orig_center_y + dy

                    # Determine appearance
                    base_color = data['color']
                    draw_color = base_color
                    if color_mode == "Matrix Color":
                        intensity = max(0.0, min(1.0, lumi)) * matrix_intensity # Use original lumi for intensity
                        draw_color = self.scale_color_brightness(matrix_rgb, intensity)

                    # Apply alpha simulation (modulate brightness)
                    if alpha_power > 0:
                        alpha_factor = lumi_scale ** alpha_power # Scale alpha based on thresholded lumi
                        draw_color = self.scale_color_brightness(draw_color, alpha_factor)

                    # Determine draw size
                    draw_size = block_size
                    if size_mode == "Variable":
                         size_factor = lumi_scale ** size_power
                         draw_size = max(min_size, int(block_size * size_factor))

                    # Calculate draw bounding box
                    half_size = draw_size / 2.0
                    x0 = int(draw_center_x - half_size)
                    y0 = int(draw_center_y - half_size)
                    x1 = int(draw_center_x + half_size)
                    y1 = int(draw_center_y + half_size)

                    # Draw (clip check is handled by PIL draw)
                    if draw_size >= 1: # Only draw if size is positive
                        draw.rectangle([x0, y0, x1, y1], fill=draw_color)

        return output_img

    # --- Helper functions t2p, p2t (remain the same) ---
    def t2p(self, t):
        if t is None or t.nelement() == 0 : return None
        try:
            img_np = np.clip(t.cpu().numpy()[0] * 255, 0, 255).astype(np.uint8)
            if img_np.shape[-1] == 1: return Image.fromarray(img_np.squeeze(), 'L')
            elif img_np.shape[-1] == 3: return Image.fromarray(img_np, 'RGB')
            elif img_np.shape[-1] == 4: return Image.fromarray(img_np, 'RGBA').convert('RGB')
            else: print(f"Warning: t2p unexpected channels: {img_np.shape[-1]}. Trying L."); return Image.fromarray(img_np.squeeze(), 'L')
        except Exception as e: print(f"Error in t2p: {e}"); return None

    def p2t(self, p):
        if p is None: return None
        try:
            img_np = np.array(p.convert('RGB')).astype(np.float32) / 255.0
            t = torch.from_numpy(img_np).unsqueeze(0).to(self.device); return t
        except Exception as e: print(f"Error in p2t: {e}"); return None


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "PixelScatterNode": PixelScatterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelScatterNode": "Pixel Scatter Effect"
}

print("--- Pixel Scatter Loaded ---")