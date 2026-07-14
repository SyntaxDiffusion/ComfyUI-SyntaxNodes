import numpy as np
import torch
from PIL import Image
import cv2
from comfy.utils import ProgressBar
from .audio_envelope_handler import AudioEnvelopeHandler

# Use relative imports to import from the current directory
from .puzzle_creator import create as create_puzzle
from .effects_handler import apply_relief_and_shadow, add_background
from .transformations_handler import transform_v1

class JigsawPuzzleNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Get standard audio inputs
        audio_inputs = AudioEnvelopeHandler.get_standard_inputs()

        return {
            "required": {
                "image": ("IMAGE",),
                "pieces": ("INT", {"default": 50, "min": 10, "max": 500, "step": 10}),
                "piece_size": ("INT", {"default": 64, "min": 32, "max": 100, "step": 1}),
                "background": ("IMAGE", {"optional": True}),
                "num_remove": ("INT", {"default": 3, "min": 0, "max": 100, "step": 1}),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Starting frame offset for audio envelope mapping"
                }),
            },
            "optional": audio_inputs
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_jigsaw_effect"
    CATEGORY = "🖼️ Image/Effects"

    def apply_jigsaw_effect(self, image, pieces, piece_size, num_remove, frame_index=0, background=None,
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
            print(f"[JigsawPuzzleNode] Mapping {batch_size} video frames to {envelope_total_frames} envelope frames (scale={frame_scale:.2f}x)")
        else:
            frame_scale = 1.0
            print(f"[JigsawPuzzleNode] Using 1:1 frame mapping (no envelope or batch_size={batch_size})")

        # Initialize list to store processed images
        processed_tensors = []

        # Process each image in the batch
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

            # Map stems to effect parameters - DIRECT mapping
            kick_val = stems['kick']
            snare_val = stems['snare']
            hihat_val = stems['hihat']

            # num_remove: MORE missing pieces on beats (kick + snare)
            combined_energy = (kick_val + snare_val) / 2.0
            remove_multiplier = 1.0 + (combined_energy * 4.0 * envelope_intensity)
            mod_num_remove = int(num_remove * remove_multiplier)

            # Clamp to valid ranges
            mod_num_remove = int(np.clip(mod_num_remove, 0, 100))

            # DEBUG: Show modulation for first few frames or when there's activity
            has_activity = kick_val > 0.1 or snare_val > 0.1 or hihat_val > 0.1
            if has_activity or idx < 3:
                print(f"[JigsawPuzzleNode] Video frame {idx} → Envelope frame {envelope_frame}:")
                print(f"  STEMS: kick={kick_val:.3f}, snare={snare_val:.3f}, hihat={hihat_val:.3f}")
                print(f"  RESULT: num_remove {num_remove}→{mod_num_remove}")

            # Convert the input image tensor to a numpy array (OpenCV-compatible)
            image_np = self.t2p(image[idx:idx+1])

            # Handle background image
            if background is not None:
                background_np = self.t2p(background[idx:idx+1] if background.shape[0] > 1 else background)
            else:
                # If no background is provided, create a white background
                background_np = np.full(image_np.shape, 255, dtype=np.uint8)

            # Create the puzzle image and puzzle mask
            puzzle_image, puzzle_mask = create_puzzle(image_np, piece_size)

            # Transform the puzzle pieces with modulated num_remove
            puzzle_image, puzzle_mask, foreground_mask = transform_v1(
                puzzle_image, puzzle_mask, piece_size, background_np.shape, mod_num_remove, select_pieces=False
            )

            # Add the background to the puzzle image
            puzzle_image = add_background(background_np, puzzle_image, foreground_mask)

            # Apply relief and shadow effects to the puzzle image
            puzzle_image, puzzle_mask = apply_relief_and_shadow(puzzle_image, puzzle_mask)

            # Convert the output puzzle image back to a tensor
            processed_tensor = self.p2t(puzzle_image)
            processed_tensors.append(processed_tensor)

            # Update progress bar
            pbar.update_absolute(idx + 1)

        # Concatenate all processed tensors along batch dimension
        final_output = torch.cat(processed_tensors, dim=0)

        return (final_output,)

    def t2p(self, t):
        """Converts a ComfyUI tensor to a NumPy array (for OpenCV)."""
        if t is not None:
            return (t.cpu().numpy().squeeze() * 255).astype(np.uint8)

    def p2t(self, p):
        """Converts a NumPy array (from OpenCV) back to a ComfyUI tensor."""
        if p is not None:
            return torch.from_numpy(p.astype(np.float32) / 255.0).unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "JigsawPuzzleNode": JigsawPuzzleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JigsawPuzzleNode": "Jigsaw Puzzle Effect"
}
