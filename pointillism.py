import numpy as np
from PIL import Image, ImageDraw
import torch
import random
from comfy.utils import ProgressBar
from .audio_envelope_handler import AudioEnvelopeHandler

class PointillismNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        # Get standard audio inputs
        audio_inputs = AudioEnvelopeHandler.get_standard_inputs()

        return {
            "required": {
                "image": ("IMAGE",),
                "dot_radius": ("INT", {"default": 3, "min": 1, "max": 10}),
                "dot_density": ("INT", {"default": 20000, "min": 1000, "max": 200000}),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Starting frame offset for audio envelope mapping"
                }),
            },
            "optional": {
                "mask": ("MASK",),
                **audio_inputs
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_pointillism"
    CATEGORY = "SyntaxNodes/Processing"

    def apply_pointillism(self, image, dot_radius, dot_density, frame_index=0, mask=None,
                         # Audio envelope parameters
                         kick_envelope="", snare_envelope="", hihat_envelope="",
                         bass_envelope="", drums_envelope="", vocals_envelope="", other_envelope="",
                         envelope_intensity=1.0, envelope_mode="multiply",
                         kick_weight=1.0, snare_weight=0.5, hihat_weight=0.3,
                         bass_weight=0.7, vocals_weight=0.5):

        # Get batch size and create progress bar
        batch_size = image.shape[0]
        pbar = ProgressBar(batch_size)

        # Get envelope duration for mapping video frames to audio frames
        envelope_total_frames = 0
        for env_str in [kick_envelope, snare_envelope, hihat_envelope, bass_envelope,
                       drums_envelope, vocals_envelope, other_envelope]:
            if env_str:
                env_data = AudioEnvelopeHandler.parse_envelope_json(env_str)
                envelope_total_frames = max(envelope_total_frames, env_data.get('total_frames', 0))

        # Calculate frame mapping: video frames → envelope frames
        if envelope_total_frames > 0 and batch_size > 0:
            frame_scale = envelope_total_frames / batch_size
            print(f"[PointillismNode] Mapping {batch_size} video frames to {envelope_total_frames} envelope frames (scale={frame_scale:.2f}x)")
        else:
            frame_scale = 1.0
            print(f"[PointillismNode] Using 1:1 frame mapping (no envelope or batch_size={batch_size})")

        # Process each image in the batch
        processed_tensors = []
        for idx in range(batch_size):
            # Map video frame to envelope frame proportionally
            envelope_frame = int(idx * frame_scale) + frame_index

            # Clamp to valid envelope range
            if envelope_total_frames > 0:
                envelope_frame = min(envelope_frame, envelope_total_frames - 1)
            envelope_frame = max(0, envelope_frame)

            # Get stem values WITHOUT adaptive processing for more variation
            stems = AudioEnvelopeHandler.get_all_stems(
                envelope_frame,
                kick_envelope, snare_envelope, hihat_envelope,
                bass_envelope, drums_envelope, vocals_envelope, other_envelope,
                adaptive=False  # Use RAW values for more dynamic range
            )

            # Map stems to effect parameters - ULTRA RESPONSIVE direct mapping
            kick_val = stems['kick']
            bass_val = stems['bass']
            hihat_val = stems['hihat']

            # dot_density: EXPLODES on kick+bass (low freq)
            # Range: base → base * (1 + 3*low_freq*intensity)
            low_freq = (kick_val + bass_val) / 2.0
            density_multiplier = 1.0 + (low_freq * 3.0 * envelope_intensity)
            mod_dot_density = int(dot_density * density_multiplier)

            # dot_radius: SPARKLES on hihat (high freq)
            # Range: base → base * (1 + 2*hihat*intensity)
            radius_multiplier = 1.0 + (hihat_val * 2.0 * envelope_intensity)
            mod_dot_radius = int(dot_radius * radius_multiplier)

            # Clamp to valid ranges
            mod_dot_density = int(np.clip(mod_dot_density, 1000, 200000))
            mod_dot_radius = int(np.clip(mod_dot_radius, 1, 10))

            # DEBUG: Show modulation for first few frames or when there's activity
            has_activity = kick_val > 0.1 or bass_val > 0.1 or hihat_val > 0.1
            if has_activity or idx < 3:
                print(f"[PointillismNode] Video frame {idx} → Envelope frame {envelope_frame}:")
                print(f"  STEMS: kick={kick_val:.3f}, bass={bass_val:.3f}, hihat={hihat_val:.3f}")
                print(f"  RESULT: density {dot_density}→{mod_dot_density}, radius {dot_radius}→{mod_dot_radius}")
            # Extract the individual image from the batch and process it
            img = image[idx:idx+1]
            
            # For handling the problematic tensor format
            img_np = img.cpu().numpy()
            if len(img_np.shape) == 4 and img_np.shape[1] == 1 and img_np.shape[3] == 3:
                # Specific handling for (1, 1, H, W, 3) format
                img_array = img_np[0, 0, :, :]  # Extract the HxWx3 array directly
                pil_img = Image.fromarray((img_array * 255).astype(np.uint8))
            else:
                # Regular conversion
                pil_img = self.t2p(img)
            
            # Ensure the image is in RGB mode
            pil_img = pil_img.convert('RGB')
            
            # Store original image for masking if needed
            if mask is not None:
                original_array = np.array(pil_img)
            
            # Apply pointillism effect with modulated parameters
            processed_image = self.generate_pointillism(pil_img, mod_dot_radius, mod_dot_density)
            
            # Apply mask if provided
            if mask is not None:
                mask_np = mask[idx].cpu().numpy()
                if len(mask_np.shape) == 2:  # Add channel dimension if needed
                    mask_np = mask_np[..., np.newaxis]
                
                # Blend original and processed images based on mask
                processed_array = np.array(processed_image)
                masked_array = original_array * (1 - mask_np) + processed_array * mask_np
                processed_image = Image.fromarray(np.clip(masked_array, 0, 255).astype(np.uint8))
            
            # Convert back to tensor
            processed_tensor = self.p2t(processed_image)
            processed_tensors.append(processed_tensor)
            
            # Update progress bar
            pbar.update_absolute(idx + 1)
        
        # Concatenate all processed tensors
        result = torch.cat(processed_tensors, dim=0)
        
        return (result,)

    def generate_pointillism(self, image, dot_radius, dot_density):
        width, height = image.size
        img_array = np.array(image)

        # Create a blank canvas
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Generate random dots based on the image colors
        for _ in range(dot_density):
            # Randomize the position of the dot
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            # Get the color of the pixel at the random position
            color = tuple(img_array[y, x])

            # Draw a circle (dot) on the canvas
            draw.ellipse(
                (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
                fill=color,
                outline=color,
            )

        return canvas

    def t2p(self, t):
        """Convert tensor to PIL Image with special handling for problematic formats."""
        if t is None:
            return None
        
        # Print shape for debugging
        print(f"Converting tensor shape: {t.shape}")
        
        # Simple approach - try direct conversion first
        try:
            img_np = t.cpu().numpy().squeeze()
            return Image.fromarray((img_np * 255).astype(np.uint8))
        except Exception as e:
            print(f"Simple conversion failed: {e}")
            
            # Specialized handling for problematic formats
            img_np = t.cpu().numpy()
            
            if len(img_np.shape) == 4 and img_np.shape[0] == 1:
                if img_np.shape[1] == 1 and img_np.shape[3] == 3:  # (1, 1, H, 3)
                    # Direct extraction of the HxWx3 array
                    img_array = img_np[0, 0]
                    return Image.fromarray((img_array * 255).astype(np.uint8))
                elif img_np.shape[1] == 3:  # (1, 3, H, W)
                    # Standard CHW to HWC conversion
                    img_array = np.transpose(img_np[0], (1, 2, 0))
                    return Image.fromarray((img_array * 255).astype(np.uint8))
            
            # If we reached here, we need more specialized handling
            raise ValueError(f"Cannot convert tensor with shape {t.shape} to PIL Image")

    def p2t(self, p):
        """Convert PIL Image to tensor."""
        if p is None:
            return None
        
        # Convert PIL to tensor
        img_np = np.array(p).astype(np.float32) / 255.0
        
        # Handle different channel arrangements
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # No need to change HWC format for ComfyUI
            pass
        elif len(img_np.shape) == 2:
            # Add channel dimension for grayscale
            img_np = img_np[..., np.newaxis]
        
        # Create tensor and add batch dimension
        return torch.from_numpy(img_np).unsqueeze(0).to(self.device)

# Register the node
NODE_CLASS_MAPPINGS = {
    "PointillismNode": PointillismNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PointillismNode": "Pointillism Effect"
}
