import numpy as np
import cv2
from PIL import Image
import torch
from comfy.utils import ProgressBar
from .audio_envelope_handler import AudioEnvelopeHandler

class RGBStreakNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Get standard audio inputs
        audio_inputs = AudioEnvelopeHandler.get_standard_inputs()

        return {
            "required": {
                "image": ("IMAGE",),
                "streak_length": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1
                }),
                "red_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "green_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "blue_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "decay": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.1,
                    "max": 0.99,
                    "step": 0.01
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
    FUNCTION = "apply_rgb_streak"
    CATEGORY = "image/effects"

    def apply_rgb_streak(self, image, streak_length, red_intensity, green_intensity, blue_intensity,
                        threshold, decay, frame_index=0,
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
            print(f"[RGBStreakNode] Mapping {batch_size} video frames to {envelope_total_frames} envelope frames (scale={frame_scale:.2f}x)")
        else:
            frame_scale = 1.0
            print(f"[RGBStreakNode] Using 1:1 frame mapping (no envelope or batch_size={batch_size})")

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

            # Map stems to effect parameters - ULTRA RESPONSIVE direct mapping
            kick_val = stems['kick']
            snare_val = stems['snare']
            hihat_val = stems['hihat']
            bass_val = stems['bass']

            # streak_length: EXTENDS on kick+bass (low freq punch)
            # Range: base → base * (1 + 2*low_freq*intensity)
            low_freq = (kick_val + bass_val) / 2.0
            length_multiplier = 1.0 + (low_freq * 2.0 * envelope_intensity)
            mod_streak_length = int(streak_length * length_multiplier)

            # red_intensity: FLASH on snare (mid freq)
            # Range: base → base * (1 + 3*snare*intensity)
            red_multiplier = 1.0 + (snare_val * 3.0 * envelope_intensity)
            mod_red_intensity = red_intensity * red_multiplier

            # green_intensity: PULSE on overall energy
            # Range: base → base * (1 + 2*energy*intensity)
            overall_energy = (kick_val + snare_val + hihat_val) / 3.0
            green_multiplier = 1.0 + (overall_energy * 2.0 * envelope_intensity)
            mod_green_intensity = green_intensity * green_multiplier

            # blue_intensity: SHIMMER on hihat (high freq)
            # Range: base → base * (1 + 2*hihat*intensity)
            blue_multiplier = 1.0 + (hihat_val * 2.0 * envelope_intensity)
            mod_blue_intensity = blue_intensity * blue_multiplier

            # Clamp to valid ranges
            mod_streak_length = int(np.clip(mod_streak_length, 1, 300))
            mod_red_intensity = float(np.clip(mod_red_intensity, 0.0, 10.0))
            mod_green_intensity = float(np.clip(mod_green_intensity, 0.0, 10.0))
            mod_blue_intensity = float(np.clip(mod_blue_intensity, 0.0, 10.0))

            # DEBUG: Show modulation for first few frames or when there's activity
            has_activity = kick_val > 0.1 or snare_val > 0.1 or hihat_val > 0.1
            if has_activity or idx < 3:
                print(f"[RGBStreakNode] Video frame {idx} → Envelope frame {envelope_frame}:")
                print(f"  STEMS: kick={kick_val:.3f}, snare={snare_val:.3f}, hihat={hihat_val:.3f}, bass={bass_val:.3f}")
                print(f"  RESULT: length {streak_length}→{mod_streak_length}, "
                      f"R {red_intensity:.2f}→{mod_red_intensity:.2f}, "
                      f"G {green_intensity:.2f}→{mod_green_intensity:.2f}, "
                      f"B {blue_intensity:.2f}→{mod_blue_intensity:.2f}")

            # Convert input tensor to numpy array
            img = image[idx].cpu().numpy()

            # Create output array (black background)
            result = np.zeros_like(img)
            height, width = img.shape[:2]

            # Process each channel with modulated intensities
            for channel_idx, intensity in enumerate([mod_red_intensity, mod_green_intensity, mod_blue_intensity]):
                channel = img[:, :, channel_idx]

                # Create mask for pixels above threshold
                mask = channel > threshold

                # Create streaks
                streak_mask = np.zeros_like(channel)

                # Direction based on channel (Red left, Green right, Blue alternating)
                if channel_idx == 0:  # Red
                    direction = -1
                elif channel_idx == 1:  # Green
                    direction = 1
                else:  # Blue
                    direction = -1

                # Apply streaking effect with modulated streak length
                for i in range(mod_streak_length):
                    # Calculate decay factor
                    current_decay = decay ** i

                    # Shift and apply intensity
                    if direction < 0:
                        shifted = np.roll(mask, -i, axis=1)
                    else:
                        shifted = np.roll(mask, i, axis=1)

                    # Add streak with decay and intensity
                    streak_contribution = channel * shifted * current_decay * intensity
                    streak_mask = np.maximum(streak_mask, streak_contribution)

                # Add random variation
                noise = np.random.normal(0, 0.02, streak_mask.shape) * (streak_mask > 0)
                streak_mask += noise

                # Add to result
                result[:, :, channel_idx] = streak_mask

            # Normalize and clip
            result = np.clip(result, 0, 1)

            # Convert back to torch tensor and append
            processed_tensors.append(torch.from_numpy(result).float().unsqueeze(0))

            # Update progress bar
            pbar.update_absolute(idx + 1)

        # Concatenate all processed tensors along batch dimension
        final_output = torch.cat(processed_tensors, dim=0)

        return (final_output,)

NODE_CLASS_MAPPINGS = {
    "RGBStreakNode": RGBStreakNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGBStreakNode": "RGB Channel Streak"
}