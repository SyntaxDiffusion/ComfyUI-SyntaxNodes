import numpy as np
import cv2
from PIL import Image
import torch
from comfy.utils import ProgressBar
from .audio_envelope_handler import AudioEnvelopeHandler

class EdgeMeasurementOverlayNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        # Get standard audio inputs
        audio_inputs = AudioEnvelopeHandler.get_standard_inputs()

        return {
            "required": {
                "image": ("IMAGE",),
                "canny_threshold1": ("FLOAT", {
                    "default": 50,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0
                }),
                "canny_threshold2": ("FLOAT", {
                    "default": 150,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0
                }),
                "min_area": ("FLOAT", {
                    "default": 100,
                    "min": 0.0,
                    "max": 10000.0,
                    "step": 1.0
                }),
                "bounding_box_opacity": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
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
    FUNCTION = "process_image"

    def process_image(self, image, canny_threshold1, canny_threshold2, min_area, bounding_box_opacity, frame_index=0,
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
            print(f"[EdgeMeasurementOverlayNode] Mapping {batch_size} video frames to {envelope_total_frames} envelope frames (scale={frame_scale:.2f}x)")
        else:
            frame_scale = 1.0
            print(f"[EdgeMeasurementOverlayNode] Using 1:1 frame mapping (no envelope or batch_size={batch_size})")

        out = []

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

            # Map stems to effect parameters using DIRECT mapping
            # Mid freq (snare) → bounding_box_opacity (flash on snare)
            snare_val = stems['snare']

            # bounding_box_opacity: ADDS on snare
            add_amount = snare_val * 0.5 * envelope_intensity
            mod_bounding_box_opacity = bounding_box_opacity + add_amount

            # Clamp to valid ranges
            mod_bounding_box_opacity = float(np.clip(mod_bounding_box_opacity, 0.0, 1.0))

            # DEBUG: Show modulation for first few frames or when there's activity
            has_activity = snare_val > 0.1
            if has_activity or idx < 3:
                print(f"[EdgeMeasurementOverlayNode] Video frame {idx} → Envelope frame {envelope_frame}:")
                print(f"  STEMS: snare={snare_val:.3f}")
                print(f"  RESULT: bounding_box_opacity {bounding_box_opacity:.2f}→{mod_bounding_box_opacity:.2f}")

            # Convert from ComfyUI image format to numpy array
            pil_image = self.t2p(image[idx:idx+1])
            frame = np.array(pil_image, dtype=np.uint8)

            # Scale up for better resolution during processing
            original_size = frame.shape[:2]
            upscale_factor = 2
            frame = cv2.resize(frame, (frame.shape[1] * upscale_factor, frame.shape[0] * upscale_factor))

            # Convert image to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Canny edge detection
            edges = cv2.Canny(gray, int(canny_threshold1), int(canny_threshold2))

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Create overlay
            overlay = np.zeros_like(frame)

            for contour in contours:
                # Filter contours by area
                area = cv2.contourArea(contour)
                if area < min_area * (upscale_factor ** 2):  # Scale threshold by the upscale factor
                    continue

                # Bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Draw styled bounding boxes
                box_color = (255, 0, 0)  # Red bounding box
                box_thickness = 2
                cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, box_thickness)

                # Add transparency effect with modulated opacity
                cv2.addWeighted(overlay, mod_bounding_box_opacity, frame, 1 - mod_bounding_box_opacity, 0, frame)

                # Add labels inside the box
                label = f"Area: {area:.2f}"
                cv2.putText(frame, label, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Scale back down to original size
            frame = cv2.resize(frame, (original_size[1], original_size[0]))

            # Convert back to PIL image
            output_image = Image.fromarray(frame)

            # Convert processed PIL image back to tensor
            out.append(self.p2t(output_image))

            # Update progress bar
            pbar.update_absolute(idx + 1)

        # Concatenate all processed tensors along batch dimension
        return (torch.cat(out, dim=0),)

    def t2p(self, t):
        if t is not None:
            # Convert tensor to PIL image
            i = 255.0 * t.cpu().numpy().squeeze()
            return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    def p2t(self, p):
        if p is not None:
            # Convert PIL image to tensor and normalize
            i = np.array(p).astype(np.float32) / 255.0
            return torch.from_numpy(i).unsqueeze(0)

NODE_CLASS_MAPPINGS = {
    "EdgeMeasurementOverlayNode": EdgeMeasurementOverlayNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EdgeMeasurementOverlayNode": "Edge Measurement Overlay"
}
