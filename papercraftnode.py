import numpy as np
import torch
from PIL import Image, ImageDraw
from comfy.utils import ProgressBar
from .audio_envelope_handler import AudioEnvelopeHandler

class PaperCraftNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        # Get standard audio inputs
        audio_inputs = AudioEnvelopeHandler.get_standard_inputs()

        return {
            "required": {
                "image": ("IMAGE",),
                "triangle_size": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 128,
                    "step": 4
                }),
                "fold_depth": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 32,
                    "step": 1
                }),
                "shadow_strength": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
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
    FUNCTION = "process_image"
    CATEGORY = "🎨 Image/Effects"

    def create_papercraft(self, image, triangle_size, fold_depth, shadow_strength):
        width, height = image.size
        result = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(result)

        # Calculate grid dimensions
        cols = width // triangle_size + 2
        rows = height // triangle_size + 2

        # Create triangular grid
        for row in range(rows):
            for col in range(cols):
                x = col * triangle_size
                y = row * triangle_size
                
                # Calculate points for two triangles that make up a square
                if (row + col) % 2 == 0:
                    points1 = [
                        (x, y),
                        (x + triangle_size, y),
                        (x, y + triangle_size)
                    ]
                    points2 = [
                        (x + triangle_size, y),
                        (x + triangle_size, y + triangle_size),
                        (x, y + triangle_size)
                    ]
                else:
                    points1 = [
                        (x, y),
                        (x + triangle_size, y),
                        (x + triangle_size, y + triangle_size)
                    ]
                    points2 = [
                        (x, y),
                        (x + triangle_size, y + triangle_size),
                        (x, y + triangle_size)
                    ]

                # Sample colors from the center of each triangle
                def get_triangle_center(points):
                    cx = sum(p[0] for p in points) // 3
                    cy = sum(p[1] for p in points) // 3
                    return (min(cx, width-1), min(cy, height-1))

                center1 = get_triangle_center(points1)
                center2 = get_triangle_center(points2)
                
                base_color1 = image.getpixel(center1)
                base_color2 = image.getpixel(center2)

                # Apply lighting effects
                light_color1 = tuple(int(c * (1 + shadow_strength)) for c in base_color1)
                light_color2 = tuple(int(c * (1 - shadow_strength)) for c in base_color2)

                # Draw main triangles
                draw.polygon(points1, fill=light_color1)
                draw.polygon(points2, fill=light_color2)

                # Add fold effects if depth is specified
                if fold_depth > 0:
                    for d in range(fold_depth):
                        shadow_factor = 1 - (d/fold_depth) * shadow_strength
                        edge_color1 = tuple(int(c * shadow_factor) for c in base_color1)
                        edge_color2 = tuple(int(c * shadow_factor) for c in base_color2)
                        
                        # Draw edges with varying shadow
                        draw.line([points1[0], points1[1]], fill=edge_color1, width=2)
                        draw.line([points1[1], points1[2]], fill=edge_color1, width=2)
                        draw.line([points1[2], points1[0]], fill=edge_color1, width=2)
                        
                        draw.line([points2[0], points2[1]], fill=edge_color2, width=2)
                        draw.line([points2[1], points2[2]], fill=edge_color2, width=2)
                        draw.line([points2[2], points2[0]], fill=edge_color2, width=2)

        return result

    def process_image(self, image, triangle_size, fold_depth, shadow_strength, frame_index=0, mask=None,
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
            print(f"[PaperCraftNode] Mapping {batch_size} video frames to {envelope_total_frames} envelope frames (scale={frame_scale:.2f}x)")
        else:
            frame_scale = 1.0
            print(f"[PaperCraftNode] Using 1:1 frame mapping (no envelope or batch_size={batch_size})")

        processed_images = []

        for b in range(batch_size):
            # Map video frame to envelope frame proportionally
            envelope_frame = int(b * frame_scale) + frame_index

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
            snare_val = stems['snare']

            # fold_depth: DEEPER FOLDS on kick (low freq)
            # Range: base → base * (1 + 4*kick*intensity)
            depth_multiplier = 1.0 + (kick_val * 4.0 * envelope_intensity)
            mod_fold_depth = int(fold_depth * depth_multiplier)

            # shadow_strength: FLASH on snare (mid freq)
            # Range: base → base + (snare*0.5*intensity)
            shadow_add = snare_val * 0.5 * envelope_intensity
            mod_shadow_strength = shadow_strength + shadow_add

            # Clamp to valid ranges
            mod_fold_depth = int(np.clip(mod_fold_depth, 0, 32))
            mod_shadow_strength = float(np.clip(mod_shadow_strength, 0.0, 1.0))

            # DEBUG: Show modulation for first few frames or when there's activity
            has_activity = kick_val > 0.1 or snare_val > 0.1
            if has_activity or b < 3:
                print(f"[PaperCraftNode] Video frame {b} → Envelope frame {envelope_frame}:")
                print(f"  STEMS: kick={kick_val:.3f}, snare={snare_val:.3f}")
                print(f"  RESULT: depth {fold_depth}→{mod_fold_depth}, shadow {shadow_strength:.2f}→{mod_shadow_strength:.2f}")

            # Convert tensor to PIL Image
            pil_image = self.tensor_to_pil(image[b:b+1])

            # Process the image with modulated parameters
            processed = self.create_papercraft(pil_image, triangle_size, mod_fold_depth, mod_shadow_strength)

            # Convert back to tensor
            processed_tensor = self.pil_to_tensor(processed)

            # Apply mask if provided
            if mask is not None:
                mask_b = mask[b:b+1] if mask is not None else None
                if mask_b is not None:
                    processed_tensor = image[b:b+1] * (1 - mask_b) + processed_tensor * mask_b

            processed_images.append(processed_tensor)
            pbar.update_absolute(b + 1)

        return (torch.cat(processed_images, dim=0),)

    def tensor_to_pil(self, tensor):
        i = 255.0 * tensor.cpu().numpy().squeeze()
        return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    def pil_to_tensor(self, pil_image):
        i = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(i).unsqueeze(0).to(self.device)

NODE_CLASS_MAPPINGS = {
    "PaperCraftNode": PaperCraftNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaperCraftNode": "Epic Paper Craft Effect"
}