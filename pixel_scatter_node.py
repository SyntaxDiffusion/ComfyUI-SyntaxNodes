import numpy as np
from PIL import Image, ImageDraw, ImageStat
import torch
from comfy.utils import ProgressBar
import random
import math


class PixelScatterNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PixelMeltNode (Point Scatter Glitch Style) using device: {self.device}")

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
                "mask": ("MASK",), # Mask applies to the final output
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "SyntaxNodes/VisualEffects" # Or Glitch

    def process_image(self, image, base_block_size, render_threshold, stochastic_drop_chance,
                      max_jitter_distance, jitter_luminance_power,
                      color_mode, matrix_color, matrix_intensity_scale,
                      size_mode, min_draw_size, size_luminance_power, alpha_luminance_power,
                      background_color, seed, mask=None):
        # --- Setup ---
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
            original_array = np.array(pil_image_rgb) if mask is not None else None

            # --- Core Point Scatter Logic ---
            processed_image = self.create_point_scatter(
                pil_image_rgb, base_block_size, bg_color_rgb,
                render_threshold, stochastic_drop_chance,
                max_jitter_distance, jitter_luminance_power,
                color_mode, matrix_color_rgb, matrix_intensity_scale,
                size_mode, min_draw_size, size_luminance_power, alpha_luminance_power
            )
            processed_array = np.array(processed_image)

            # --- Masking (Standard) ---
            final_array = processed_array
            if mask is not None and idx < mask.shape[0]:
                # ... (same masking logic as previous versions) ...
                single_mask_tensor = mask[idx:idx+1]
                if single_mask_tensor.shape[1] == H and single_mask_tensor.shape[2] == W:
                    mask_array = single_mask_tensor.squeeze().cpu().numpy()
                    if len(mask_array.shape) == 2: mask_array = mask_array[..., np.newaxis]
                    if original_array is not None and original_array.shape[:2] == mask_array.shape[:2]:
                         final_array = original_array * (1 - mask_array) + processed_array * mask_array
                         final_array = final_array.astype(np.uint8)
                    else:
                        if original_array is None: final_array = (processed_array * mask_array).astype(np.uint8)
                        else: print(f"Mask/Image shape mismatch idx {idx}. Skipping."); final_array = processed_array
                else: print(f"Mask dimensions mismatch idx {idx}. Skipping."); final_array = processed_array

            # --- Output ---
            final_image_pil = Image.fromarray(final_array)
            processed_tensor = self.p2t(final_image_pil)
            if processed_tensor is not None: processed_tensors.append(processed_tensor)
            pbar.update_absolute(idx + 1)

        # --- Batch Concat ---
        if not processed_tensors: print("Error: No images processed."); return (image,)
        device = processed_tensors[0].device
        tensors_on_device = [t.to(device) for t in processed_tensors]
        try: final_output = torch.cat(tensors_on_device, dim=0)
        except Exception as e: print(f"Error concatenating: {e}"); return(image,)
        return (final_output,)

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

    # --- Core Logic ---
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
                    # Use ellipse for more point-cloud like feel? Or keep rectangle?
                    # Let's use rectangle for consistency with blocks, size variation helps.
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