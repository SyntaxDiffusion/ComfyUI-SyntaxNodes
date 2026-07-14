import numpy as np
import torch
from PIL import Image
import cv2
from comfy.utils import ProgressBar
from .audio_envelope_handler import AudioEnvelopeHandler


class EdgeTracingNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        # Get standard audio inputs
        audio_inputs = AudioEnvelopeHandler.get_standard_inputs()

        return {
            "required": {
                "input_image": ("IMAGE",),
                "low_threshold": ("INT", {"default": 50, "min": 0, "max": 255}),
                "high_threshold": ("INT", {"default": 150, "min": 0, "max": 255}),
                "num_particles": ("INT", {"default": 1000, "min": 1, "max": 50000}),
                "speed": ("INT", {"default": 10, "min": 1, "max": 100}),
                "edge_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "particle_size": ("INT", {"default": 1, "min": 1, "max": 10}),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Starting frame offset for audio envelope mapping"
                }),
            },
            "optional": audio_inputs
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_particle_tracing"

    def generate_particle_tracing(
        self, input_image, low_threshold, high_threshold, num_particles, speed, edge_opacity, particle_size, frame_index=0,
        # Audio envelope parameters
        kick_envelope="", snare_envelope="", hihat_envelope="",
        bass_envelope="", drums_envelope="", vocals_envelope="", other_envelope="",
        envelope_intensity=1.0, envelope_mode="multiply",
        kick_weight=1.0, snare_weight=0.5, hihat_weight=0.3,
        bass_weight=0.7, vocals_weight=0.5):

        # Get batch size and create progress bar
        batch_size = input_image.shape[0]
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
            print(f"[EdgeTracingNode] Mapping {batch_size} video frames to {envelope_total_frames} envelope frames (scale={frame_scale:.2f}x)")
        else:
            frame_scale = 1.0
            print(f"[EdgeTracingNode] Using 1:1 frame mapping (no envelope or batch_size={batch_size})")

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
            # Low freq (kick) → num_particles (more particles on kick)
            kick_val = stems['kick']
            # High freq (hihat) → speed (faster on hihat)
            hihat_val = stems['hihat']

            # num_particles: MULTIPLIES on kick
            multiplier = 1.0 + (kick_val * 3.0 * envelope_intensity)
            mod_num_particles = int(num_particles * multiplier)

            # speed: MULTIPLIES on hihat
            speed_multiplier = 1.0 + (hihat_val * 4.0 * envelope_intensity)
            mod_speed = int(speed * speed_multiplier)

            # Clamp to valid ranges
            mod_num_particles = int(np.clip(mod_num_particles, 1, 50000))
            mod_speed = int(np.clip(mod_speed, 1, 100))

            # DEBUG: Show modulation for first few frames or when there's activity
            has_activity = kick_val > 0.1 or hihat_val > 0.1
            if has_activity or idx < 3:
                print(f"[EdgeTracingNode] Video frame {idx} → Envelope frame {envelope_frame}:")
                print(f"  STEMS: kick={kick_val:.3f}, hihat={hihat_val:.3f}")
                print(f"  RESULT: num_particles {num_particles}→{mod_num_particles}, speed {speed}→{mod_speed}")

            # Step 1: Convert input tensor to grayscale numpy image
            pil_image = self.t2p(input_image[idx:idx+1])
            np_image = np.array(pil_image.convert("L"))

            # Step 2: Apply Canny edge detection using OpenCV
            edges = cv2.Canny(np_image, threshold1=low_threshold, threshold2=high_threshold)
            edge_coords = np.column_stack(np.where(edges > 0))

            if edge_coords.shape[0] == 0:
                raise ValueError("No edges detected. Adjust thresholds.")

            # Step 3: Initialize particles at random edge coordinates with modulated count
            particle_positions = edge_coords[
                np.random.choice(edge_coords.shape[0], mod_num_particles, replace=True)
            ]

            # Step 4: Precompute 8-neighbor offsets
            neighbor_offsets = np.array(
                [
                    [-1, -1], [-1, 0], [-1, 1],
                    [0, -1],          [0, 1],
                    [1, -1], [1, 0], [1, 1]
                ]
            )

            # Step 5: Create separate canvases for edges and particles
            edge_canvas = edges.astype(np.float32) * edge_opacity
            particle_canvas = np.zeros_like(edges, dtype=np.float32)

            for _ in range(mod_speed):  # Run particle tracing for the specified number of steps with modulated speed
                # Calculate neighbors
                neighbors = particle_positions[:, None, :] + neighbor_offsets[None, :, :]
                valid_mask = (
                    (neighbors[:, :, 0] >= 0) & (neighbors[:, :, 1] >= 0) &
                    (neighbors[:, :, 0] < edges.shape[0]) & (neighbors[:, :, 1] < edges.shape[1])
                )
                neighbors = neighbors[valid_mask]

                # Keep only neighbors that are edge pixels
                valid_neighbors = neighbors[edges[neighbors[:, 0], neighbors[:, 1]] > 0]

                if valid_neighbors.shape[0] > 0:
                    particle_positions = valid_neighbors[
                        np.random.choice(valid_neighbors.shape[0], mod_num_particles, replace=True)
                    ]

                # Draw particles onto the particle canvas
                for y, x in particle_positions:
                    cv2.circle(particle_canvas, (x, y), particle_size, 1, thickness=-1)

            # Normalize the particle canvas for visibility
            particle_canvas = (particle_canvas / particle_canvas.max() * 255).clip(0, 255).astype(np.uint8)

            # Step 6: Combine edge and particle layers
            combined_canvas = np.stack(
                [particle_canvas, edge_canvas.astype(np.uint8), np.zeros_like(edge_canvas)], axis=-1
            ).astype(np.uint8)

            # Step 7: Convert combined canvas to tensor and append
            combined_image = Image.fromarray(combined_canvas, "RGB")
            out.append(self.p2t(combined_image))

            # Update progress bar
            pbar.update_absolute(idx + 1)

        # Concatenate all processed tensors along batch dimension
        return (torch.cat(out, dim=0),)

    def t2p(self, tensor):
        """Convert a tensor to a PIL image."""
        if tensor is not None:
            i = 255.0 * tensor.cpu().numpy().squeeze()
            if len(i.shape) == 3 and i.shape[0] == 1:  # Handle extra channel dimension
                i = i[0]
            return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    def p2t(self, pil_image):
        """Convert a PIL image to a tensor."""
        if pil_image is not None:
            np_image = np.array(pil_image).astype(np.float32) / 255.0
            return torch.from_numpy(np_image).unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "EdgeTracingNode": EdgeTracingNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EdgeTracingNode": "Edge Tracing Node"
}
