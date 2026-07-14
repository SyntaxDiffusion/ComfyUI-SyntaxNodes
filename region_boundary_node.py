import numpy as np
import torch
from PIL import Image
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from skimage.segmentation import slic, mark_boundaries, watershed
    from skimage.filters import sobel
    from skimage.color import rgb2gray
    from comfy.utils import ProgressBar
except ImportError:
    print("Installing scikit-image...")
    install_package("scikit-image")
    from skimage.segmentation import slic, mark_boundaries, watershed
    from skimage.filters import sobel
    from skimage.color import rgb2gray

from .audio_envelope_handler import AudioEnvelopeHandler

class RegionBoundaryNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Get standard audio inputs
        audio_inputs = AudioEnvelopeHandler.get_standard_inputs()

        return {
            "required": {
                "image": ("IMAGE",),
                "segments": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
                "compactness": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                "line_color": ("INT", {"default": 0xFFFFFF, "min": 0, "max": 0xFFFFFF, "step": 1}),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Starting frame offset for audio envelope mapping"
                }),
            },
            "optional": audio_inputs
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_region_boundary"
    CATEGORY = "SyntaxNodes/Processing"

    def apply_region_boundary(self, image, segments, compactness, line_color, frame_index=0,
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
            print(f"[RegionBoundaryNode] Mapping {batch_size} video frames to {envelope_total_frames} envelope frames (scale={frame_scale:.2f}x)")
        else:
            frame_scale = 1.0
            print(f"[RegionBoundaryNode] Using 1:1 frame mapping (no envelope or batch_size={batch_size})")

        # Initialize list to store processed images
        result = []

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

            # segments: EXPLODES on overall energy (more fragmentation on beats)
            overall_energy = (kick_val + snare_val + hihat_val) / 3.0
            segments_multiplier = 1.0 + (overall_energy * 5.0 * envelope_intensity)
            mod_segments = int(segments * segments_multiplier)

            # compactness: Varies with hihat (tighter regions on hi-hats)
            compactness_add = hihat_val * 20.0 * envelope_intensity
            mod_compactness = compactness + compactness_add

            # Clamp to valid ranges
            mod_segments = int(np.clip(mod_segments, 10, 1000))
            mod_compactness = float(np.clip(mod_compactness, 1.0, 100.0))

            # DEBUG: Show modulation for first few frames or when there's activity
            has_activity = kick_val > 0.1 or snare_val > 0.1 or hihat_val > 0.1
            if has_activity or idx < 3:
                print(f"[RegionBoundaryNode] Video frame {idx} → Envelope frame {envelope_frame}:")
                print(f"  STEMS: kick={kick_val:.3f}, snare={snare_val:.3f}, hihat={hihat_val:.3f}")
                print(f"  RESULT: segments {segments}→{mod_segments}, compactness {compactness:.1f}→{mod_compactness:.1f}")

            # Convert to numpy array
            img_np = np.array(self.t2p(image[idx]))

            # Apply SLIC segmentation with modulated segments and compactness
            segments_slic = slic(img_np, n_segments=mod_segments, compactness=mod_compactness, start_label=1)

            # Use watershed for additional refinement
            gradient = sobel(rgb2gray(img_np))
            labels = watershed(gradient, segments_slic)

            # Draw region boundaries
            line_color_rgb = self.int_to_rgb(line_color)
            boundary_image = mark_boundaries(img_np, labels, color=line_color_rgb)

            # Convert back to tensor
            result.append(self.p2t(Image.fromarray((boundary_image * 255).astype(np.uint8))))

            # Update progress bar
            pbar.update_absolute(idx + 1)

        return (torch.cat(result, dim=0),)

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            t = torch.from_numpy(i).unsqueeze(0)
        return t

    def int_to_rgb(self, color_int):
        return ((color_int >> 16) & 255) / 255.0, ((color_int >> 8) & 255) / 255.0, (color_int & 255) / 255.0

NODE_CLASS_MAPPINGS = {
    "RegionBoundaryNode": RegionBoundaryNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionBoundaryNode": "Region Boundary Effect"
}