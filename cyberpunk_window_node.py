import numpy as np
import cv2
from PIL import Image
import torch
import random
from comfy.utils import ProgressBar
from .audio_envelope_handler import AudioEnvelopeHandler

class CyberpunkWindowNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        # Get standard audio inputs
        audio_inputs = AudioEnvelopeHandler.get_standard_inputs()

        return {
            "required": {
                "image": ("IMAGE",),
                "custom_text": ("STRING", {"default": "MOVEMENT", "multiline": False}),
                "edge_threshold1": ("FLOAT", {
                    "default": 100,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0
                }),
                "edge_threshold2": ("FLOAT", {
                    "default": 200,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0
                }),
                "min_window_size": ("FLOAT", {
                    "default": 50,
                    "min": 1.0,
                    "max": 1000.0,
                    "step": 10.0
                }),
                "max_windows": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 500,
                    "step": 1
                }),
                "line_thickness": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "glow_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "text_size": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "preserve_background": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
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
    FUNCTION = "create_cyberpunk_effect"
    CATEGORY = "SyntaxNodes/Processing"

    def process_single_image(self, image_tensor, edge_threshold1, edge_threshold2,
                           min_window_size, max_windows, line_thickness,
                           glow_intensity, text_size, preserve_background, custom_text,
                           mod_glow_intensity, mod_max_windows):
        # Convert from tensor to PIL
        pil_image = self.t2p(image_tensor)
        frame = np.array(pil_image)

        # Create background based on preserve_background flag
        if preserve_background:
            result = frame.copy()
        else:
            result = np.zeros_like(frame)

        # Edge detection for finding potential windows
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, edge_threshold1, edge_threshold2)

        # Find contours for windows
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area and keep only the largest ones (use modulated max_windows)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:mod_max_windows]

        # Store window centers for connecting lines
        window_centers = []

        # Process each window
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_window_size:
                continue

            # Create window mask
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            window_centers.append(center)

            if not preserve_background:
                # Copy original image content for this window
                mask = np.zeros_like(frame)
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
                window_content = cv2.bitwise_and(frame, mask)

                # Add window content to result
                result = cv2.add(result, window_content)

            # Add glowing border effect (use modulated glow intensity)
            glow_color = (
                random.randint(150, 255),  # R
                random.randint(150, 255),  # G
                random.randint(150, 255)   # B
            )

            # Multiple borders with decreasing intensity for glow effect
            for i in range(3):
                thickness = line_thickness + i*2
                alpha = mod_glow_intensity * (1 - i*0.2)
                glow = np.zeros_like(frame)
                cv2.rectangle(glow, (x-i*2, y-i*2), (x + w+i*2, y + h+i*2),
                            glow_color, thickness)
                result = cv2.addWeighted(result, 1, glow, alpha, 0)

            # Add measurement text with custom prefix
            label = f"{custom_text}: {area:.0f}px"
            font_scale = text_size
            font_thickness = max(1, int(line_thickness * 0.8))

            # Get text size for better positioning
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Position text inside the window near the top
            text_x = x + 5
            text_y = y + text_height + 5

            # Add black background for text readability
            cv2.rectangle(result,
                         (text_x - 2, text_y - text_height - 2),
                         (text_x + text_width + 2, text_y + 2),
                         (0, 0, 0), -1)

            # Draw text
            cv2.putText(result, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (0, 255, 0), font_thickness)

        # Connect windows with RGB lines
        if len(window_centers) > 1:
            for i in range(len(window_centers)-1):
                start = window_centers[i]
                end = window_centers[i+1]

                # Create RGB offset lines
                offset = 2
                cv2.line(result,
                        (start[0]-offset, start[1]),
                        (end[0]-offset, end[1]),
                        (255, 0, 0),
                        1)  # Red
                cv2.line(result,
                        start,
                        end,
                        (0, 255, 0),
                        1)  # Green
                cv2.line(result,
                        (start[0]+offset, start[1]),
                        (end[0]+offset, end[1]),
                        (0, 0, 255),
                        1)  # Blue

        # Convert back to tensor
        output_image = Image.fromarray(result)
        return self.p2t(output_image)

    def create_cyberpunk_effect(self, image, custom_text, edge_threshold1, edge_threshold2,
                              min_window_size, max_windows, line_thickness,
                              glow_intensity, text_size, preserve_background, frame_index=0,
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
            print(f"[CyberpunkWindowNode] Mapping {batch_size} video frames to {envelope_total_frames} envelope frames (scale={frame_scale:.2f}x)")
        else:
            frame_scale = 1.0
            print(f"[CyberpunkWindowNode] Using 1:1 frame mapping (no envelope or batch_size={batch_size})")

        # Process each image in the batch
        processed_images = []
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

            # Map stems to effect parameters - Direct mapping
            # Mid freq (snare) → glow_intensity (flash on snare)
            snare_val = stems['snare']
            glow_multiplier = 1.0 + (snare_val * 3.0 * envelope_intensity)
            mod_glow_intensity = glow_intensity * glow_multiplier

            # Overall energy (kick + snare + hihat) → max_windows
            kick_val = stems['kick']
            hihat_val = stems['hihat']
            overall_energy = (kick_val + snare_val + hihat_val) / 3.0
            windows_multiplier = 1.0 + (overall_energy * 2.0 * envelope_intensity)
            mod_max_windows = int(max_windows * windows_multiplier)

            # Ensure valid ranges
            mod_glow_intensity = float(np.clip(mod_glow_intensity, 0.0, 10.0))
            mod_max_windows = int(np.clip(mod_max_windows, 1, 500))

            # DEBUG: Show modulation for first few frames or when there's activity
            has_activity = kick_val > 0.1 or snare_val > 0.1
            if has_activity or idx < 3:
                print(f"[CyberpunkWindowNode] Video frame {idx} → Envelope frame {envelope_frame}:")
                print(f"  STEMS: kick={kick_val:.3f}, snare={snare_val:.3f}, hihat={hihat_val:.3f}")
                print(f"  RESULT: glow {glow_intensity:.2f}→{mod_glow_intensity:.2f}, windows {max_windows}→{mod_max_windows}")

            # Process single image
            processed = self.process_single_image(
                image[idx:idx+1],
                edge_threshold1,
                edge_threshold2,
                min_window_size,
                max_windows,
                line_thickness,
                glow_intensity,
                text_size,
                preserve_background,
                custom_text,
                mod_glow_intensity,
                mod_max_windows
            )
            processed_images.append(processed)
            pbar.update_absolute(idx + 1)

        # Concatenate results and return
        return (torch.cat(processed_images, dim=0),)

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            return torch.from_numpy(i).unsqueeze(0)

NODE_CLASS_MAPPINGS = {
    "CyberpunkWindowNode": CyberpunkWindowNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CyberpunkWindowNode": "Cyberpunk Window Effect"
}
