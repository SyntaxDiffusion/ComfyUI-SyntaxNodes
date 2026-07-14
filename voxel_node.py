import numpy as np
from PIL import Image, ImageDraw
import torch
from comfy.utils import ProgressBar
from .audio_envelope_handler import AudioEnvelopeHandler

class VoxelNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        # Get standard audio inputs
        audio_inputs = AudioEnvelopeHandler.get_standard_inputs()

        return {
            "required": {
                "image": ("IMAGE",),
                "block_size": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "step": 1
                }),
                "block_depth": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 32,
                    "step": 1
                }),
                "shading": ("FLOAT", {
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
    CATEGORY = "SyntaxNodes/Processing"
    
    def process_image(self, image, block_size, block_depth, shading, frame_index=0, mask=None,
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
        # Check all envelope inputs and use the maximum total_frames
        envelope_total_frames = 0
        for env_str in [kick_envelope, snare_envelope, hihat_envelope, bass_envelope,
                       drums_envelope, vocals_envelope, other_envelope]:
            if env_str:
                env_data = AudioEnvelopeHandler.parse_envelope_json(env_str)
                envelope_total_frames = max(envelope_total_frames, env_data.get('total_frames', 0))

        # Calculate frame mapping: video frames → envelope frames
        # If envelope exists and has more frames than video, map proportionally
        if envelope_total_frames > 0 and batch_size > 0:
            frame_scale = envelope_total_frames / batch_size
            print(f"[VoxelNode] Mapping {batch_size} video frames to {envelope_total_frames} envelope frames (scale={frame_scale:.2f}x)")
        else:
            frame_scale = 1.0
            print(f"[VoxelNode] Using 1:1 frame mapping (no envelope or batch_size={batch_size})")

        # Initialize list to store processed images
        processed_tensors = []

        # Smoothing state (retains values across frames for smooth transitions)
        smooth_block_size = float(block_size)
        smooth_block_depth = float(block_depth)
        smooth_shading = float(shading)

        # Process each image in the batch with its corresponding frame
        for idx in range(batch_size):
            # Map video frame to envelope frame proportionally
            # frame_index is added AFTER scaling (not before, to avoid overflow)
            envelope_frame = int(idx * frame_scale) + frame_index

            # Clamp to valid envelope range
            if envelope_total_frames > 0:
                envelope_frame = min(envelope_frame, envelope_total_frames - 1)
            envelope_frame = max(0, envelope_frame)

            # Get stem values WITHOUT adaptive processing first (to see raw values)
            stems_raw = AudioEnvelopeHandler.get_all_stems(
                envelope_frame,
                kick_envelope, snare_envelope, hihat_envelope,
                bass_envelope, drums_envelope, vocals_envelope, other_envelope,
                adaptive=False
            )

            # Get stem values WITHOUT adaptive processing for more variation
            stems = AudioEnvelopeHandler.get_all_stems(
                envelope_frame,
                kick_envelope, snare_envelope, hihat_envelope,
                bass_envelope, drums_envelope, vocals_envelope, other_envelope,
                adaptive=False  # Use RAW values for more dynamic range
            )

            # DEBUG: Log first few frames and any frames with activity
            has_activity = any(v > 0.1 for v in stems_raw.values())
            if has_activity or idx < 5:
                print(f"[VoxelNode] Video frame {idx} → Envelope frame {envelope_frame}:")
                print(f"  RAW:       kick={stems_raw['kick']:.3f}, snare={stems_raw['snare']:.3f}, "
                      f"hihat={stems_raw['hihat']:.3f}, bass={stems_raw['bass']:.3f}")
                print(f"  ADAPTIVE:  kick={stems['kick']:.3f}, snare={stems['snare']:.3f}, "
                      f"hihat={stems['hihat']:.3f}, bass={stems['bass']:.3f}")

            # Map stems to effect parameters - USE RAW VALUES (no weights for cleaner response)
            kick_val = stems['kick']
            snare_val = stems['snare']

            # ULTRA RESPONSIVE: Direct mapping, no smoothing, no thresholds
            # Just use the RAW audio values directly!

            # block_size: EXPLODES on kick (proportional to kick strength)
            # Range: base → base * (1 + 4*kick*intensity)
            size_multiplier = 1.0 + (kick_val * 4.0 * envelope_intensity)
            mod_block_size = int(block_size * size_multiplier)

            # block_depth: Dramatic 3D pop on kicks
            # Range: base → base + (kick * 24 * intensity)
            depth_add = kick_val * 24 * envelope_intensity
            mod_block_depth = int(block_depth + depth_add)

            # shading: Flash on snares
            shade_add = snare_val * 0.5 * envelope_intensity
            mod_shading = shading + shade_add

            # Clamp to valid ranges
            mod_block_size = int(np.clip(mod_block_size, 4, 64))
            mod_block_depth = int(np.clip(mod_block_depth, 0, 32))
            mod_shading = float(np.clip(mod_shading, 0.0, 1.0))

            # Mark strong beats for debug
            kick_trigger = 1.0 if kick_val > 0.7 else 0.0
            snare_trigger = 1.0 if snare_val > 0.7 else 0.0

            # DEBUG: Show modulation results when there's activity or trigger
            if has_activity or idx < 5 or kick_trigger > 0 or snare_trigger > 0:
                trigger_str = ""
                if kick_trigger > 0:
                    trigger_str += "🥁KICK "
                if snare_trigger > 0:
                    trigger_str += "🎵SNARE "

                print(f"  RESPONSIVE: kick={kick_val:.3f}, snare={snare_val:.3f} {trigger_str}")
                print(f"  MULTIPLIER: size×{size_multiplier:.2f}, depth+{depth_add:.1f}")
                print(f"  RESULT: block_size {block_size}→{mod_block_size}, "
                      f"depth {block_depth}→{mod_block_depth}, "
                      f"shading {shading:.2f}→{mod_shading:.2f}")

            # Extract single image from batch
            single_image = image[idx:idx+1]

            # Convert from ComfyUI image format to PIL
            pil_image = self.t2p(single_image)

            # Ensure the image is in RGB mode
            pil_image = pil_image.convert('RGB')
            original_array = np.array(pil_image)

            # Process the image with modulated parameters
            processed_image = self.create_voxel(pil_image, mod_block_size, mod_block_depth, mod_shading)
            processed_array = np.array(processed_image)
            
            # Handle masking for single image
            if mask is not None:
                mask_array = mask[idx:idx+1].squeeze().cpu().numpy()
                # Ensure mask has same dimensions as image
                if len(mask_array.shape) == 2:
                    mask_array = mask_array[..., np.newaxis]
                # Apply mask
                masked_array = original_array * (1 - mask_array) + processed_array * mask_array
                processed_image = Image.fromarray(masked_array.astype(np.uint8))
            
            # Convert back to tensor and append to list
            processed_tensor = self.p2t(processed_image)
            processed_tensors.append(processed_tensor)
            
            # Update progress bar
            pbar.update_absolute(idx + 1)
        
        # Concatenate all processed tensors along batch dimension
        final_output = torch.cat(processed_tensors, dim=0)
        
        return (final_output,)

    def create_voxel(self, image, block_size, block_depth, shading):
        width, height = image.size
        
        # Create a new image with the same size
        result = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(result)
        
        # Calculate grid dimensions
        cols = width // block_size + (1 if width % block_size else 0)
        rows = height // block_size + (1 if height % block_size else 0)
        
        # Create blocks
        for row in range(rows):
            for col in range(cols):
                # Calculate block position
                x = col * block_size
                y = row * block_size
                
                # Sample color from original image
                sample_x = min(x + block_size//2, width-1)
                sample_y = min(y + block_size//2, height-1)
                base_color = image.getpixel((sample_x, sample_y))
                
                # Draw main face of block
                block_points = [
                    (x, y),
                    (x + block_size, y),
                    (x + block_size, y + block_size),
                    (x, y + block_size)
                ]
                draw.polygon(block_points, fill=base_color)
                
                if block_depth > 0:
                    # Calculate offset for 3D effect
                    offset = block_depth
                    
                    # Top face (if visible)
                    if y > 0:
                        top_color = tuple(int(c * (1 + shading)) for c in base_color)
                        top_points = [
                            (x, y),
                            (x + offset, y - offset),
                            (x + block_size + offset, y - offset),
                            (x + block_size, y)
                        ]
                        draw.polygon(top_points, fill=top_color)
                    
                    # Right face
                    right_color = tuple(int(c * (1 - shading)) for c in base_color)
                    right_points = [
                        (x + block_size, y),
                        (x + block_size + offset, y - offset),
                        (x + block_size + offset, y + block_size - offset),
                        (x + block_size, y + block_size)
                    ]
                    draw.polygon(right_points, fill=right_color)
        
        return result

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            t = torch.from_numpy(i).unsqueeze(0).to(self.device)
        return t

NODE_CLASS_MAPPINGS = {
    "VoxelNode": VoxelNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoxelNode": "Voxel Block Effect"
}