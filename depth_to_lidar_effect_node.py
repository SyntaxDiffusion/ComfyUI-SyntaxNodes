import cv2
import numpy as np
import torch
from PIL import Image
from comfy.utils import ProgressBar
from .audio_envelope_handler import AudioEnvelopeHandler

class DepthToLidarEffectNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prev_frame = None  # Maintain single previous frame for temporal consistency

    @classmethod
    def INPUT_TYPES(cls):
        # Get standard audio inputs
        audio_inputs = AudioEnvelopeHandler.get_standard_inputs()

        return {
            "required": {
                "depth_map": ("IMAGE",),
                "smoothing_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "line_thickness": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1
                }),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Starting frame offset for audio envelope mapping"
                }),
            },
            "optional": audio_inputs
        }

    RETURN_TYPES = ("IMAGE",)
    # Fix: Make FUNCTION match the actual method name
    FUNCTION = "process_depth_map"
    CATEGORY = "SyntaxNodes/Processing"
    
    def process_depth_map(self, depth_map, smoothing_factor, line_thickness, frame_index=0,
                          # Audio envelope parameters
                          kick_envelope="", snare_envelope="", hihat_envelope="",
                          bass_envelope="", drums_envelope="", vocals_envelope="", other_envelope="",
                          envelope_intensity=1.0, envelope_mode="multiply",
                          kick_weight=1.0, snare_weight=0.5, hihat_weight=0.3,
                          bass_weight=0.7, vocals_weight=0.5):

        # Get batch size and create progress bar
        batch_size = depth_map.shape[0]
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
            print(f"[DepthToLidarEffectNode] Mapping {batch_size} video frames to {envelope_total_frames} envelope frames (scale={frame_scale:.2f}x)")
        else:
            frame_scale = 1.0
            print(f"[DepthToLidarEffectNode] Using 1:1 frame mapping (no envelope or batch_size={batch_size})")

        out = []

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

            # Map stems to effect parameters using DIRECT mapping
            # Mid freq (snare) → line_thickness (thicker lines on snare)
            snare_val = stems['snare']

            # line_thickness: MULTIPLIES on snare
            multiplier = 1.0 + (snare_val * 3.0 * envelope_intensity)
            mod_line_thickness = int(line_thickness * multiplier)

            # Clamp to valid ranges
            mod_line_thickness = int(np.clip(mod_line_thickness, 1, 5))

            # DEBUG: Show modulation for first few frames or when there's activity
            has_activity = snare_val > 0.1
            if has_activity or b < 3:
                print(f"[DepthToLidarEffectNode] Video frame {b} → Envelope frame {envelope_frame}:")
                print(f"  STEMS: snare={snare_val:.3f}")
                print(f"  RESULT: line_thickness {line_thickness}→{mod_line_thickness}")
            # Convert depth map tensor to a PIL image
            depth_image = self.t2p(depth_map[b:b+1])
            depth_array = np.array(depth_image)

            # Normalize and process the depth map
            depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_blurred = cv2.GaussianBlur(depth_normalized, (5, 5), 0)
            edges = cv2.Canny(depth_blurred, 50, 150)

            # Check if prev_frame exists and matches the current frame dimensions
            if self.prev_frame is not None and self.prev_frame.shape != edges.shape:
                self.prev_frame = None  # Reset if sizes differ

            # Temporal smoothing with previous frame
            if self.prev_frame is not None:
                edges = cv2.addWeighted(edges, smoothing_factor, self.prev_frame, 1 - smoothing_factor, 0)
            
            # Update the previous frame
            self.prev_frame = edges.copy()

            # Convert edges to white lines on a black background
            output = np.zeros_like(depth_array)
            output[edges > 0] = 255

            # Convert processed image back to tensor
            output_image = Image.fromarray(output)
            output_tensor = self.p2t(output_image)
            out.append(output_tensor)

            # Update progress bar
            pbar.update_absolute(b + 1)

        return (torch.cat(out, dim=0),)

    def reset(self):
        """Reset the previous frame memory for a new sequence."""
        self.prev_frame = None

    def t2p(self, t):
        # Convert ComfyUI tensor format to PIL image
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def p2t(self, p):
        # Convert PIL image to ComfyUI tensor format and move to GPU if available
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            t = torch.from_numpy(i).unsqueeze(0).to(self.device)
        return t

NODE_CLASS_MAPPINGS = {
    "DepthToLidarEffectNode": DepthToLidarEffectNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthToLidarEffectNode": "Depth to LIDAR Effect"
}