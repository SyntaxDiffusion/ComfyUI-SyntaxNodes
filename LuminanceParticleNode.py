import cv2
import numpy as np
import torch
from PIL import Image
from comfy.utils import ProgressBar
import random
from .audio_envelope_handler import AudioEnvelopeHandler


class LuminanceParticleNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prev_frame = None
        self.particles = []  # List to store particles

    @classmethod
    def INPUT_TYPES(cls):
        # Get standard audio inputs
        audio_inputs = AudioEnvelopeHandler.get_standard_inputs()

        return {
            "required": {
                "depth_map": ("IMAGE",),
                "num_layers": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "step": 1
                }),
                "smoothing_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "particle_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "particle_speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                "num_particles": ("INT", {
                    "default": 200,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "particle_opacity": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "edge_opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "particle_lifespan": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
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
    FUNCTION = "process_depth_map"

    def process_depth_map(self, depth_map, num_layers, smoothing_factor, particle_size, particle_speed, num_particles, particle_opacity, edge_opacity, particle_lifespan, frame_index=0,
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
            print(f"[LuminanceParticleNode] Mapping {batch_size} video frames to {envelope_total_frames} envelope frames (scale={frame_scale:.2f}x)")
        else:
            frame_scale = 1.0
            print(f"[LuminanceParticleNode] Using 1:1 frame mapping (no envelope or batch_size={batch_size})")

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

            # num_particles: EXPLODES on kick
            particles_multiplier = 1.0 + (kick_val * 3.0 * envelope_intensity)
            mod_num_particles = int(num_particles * particles_multiplier)

            # particle_speed: Accelerates on hihat
            speed_multiplier = 1.0 + (hihat_val * 2.0 * envelope_intensity)
            mod_particle_speed = particle_speed * speed_multiplier

            # particle_opacity: Pulses on snare
            opacity_add = snare_val * 0.3 * envelope_intensity
            mod_particle_opacity = particle_opacity + opacity_add

            # Clamp to valid ranges
            mod_num_particles = int(np.clip(mod_num_particles, 1, 1000))
            mod_particle_speed = float(np.clip(mod_particle_speed, 0.1, 5.0))
            mod_particle_opacity = float(np.clip(mod_particle_opacity, 0.0, 1.0))

            # DEBUG: Show modulation for first few frames or when there's activity
            has_activity = kick_val > 0.1 or snare_val > 0.1 or hihat_val > 0.1
            if has_activity or idx < 3:
                print(f"[LuminanceParticleNode] Video frame {idx} → Envelope frame {envelope_frame}:")
                print(f"  STEMS: kick={kick_val:.3f}, snare={snare_val:.3f}, hihat={hihat_val:.3f}")
                print(f"  RESULT: num_particles {num_particles}→{mod_num_particles}, speed {particle_speed:.2f}→{mod_particle_speed:.2f}, opacity {particle_opacity:.2f}→{mod_particle_opacity:.2f}")

            # Convert depth map tensor to a PIL image and then to a NumPy array
            depth_image = self.t2p(depth_map[idx:idx+1])
            depth_array = np.array(depth_image)

            # Normalize depth array
            depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Compute gradients for directional flow and transfer them to the GPU
            grad_x = cv2.Sobel(depth_normalized, cv2.CV_32F, 1, 0, ksize=5)
            grad_y = cv2.Sobel(depth_normalized, cv2.CV_32F, 0, 1, ksize=5)
            magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
            angle = torch.from_numpy(angle).to(self.device, dtype=torch.float32)

            # Detect edges
            edges = cv2.Canny(depth_normalized, 50, 150)

            # Transfer edge image to the GPU
            edge_layer = torch.zeros((depth_normalized.shape[0], depth_normalized.shape[1], 3), device=self.device, dtype=torch.float32)
            edge_layer[torch.from_numpy(edges > 0).to(self.device, dtype=torch.bool)] = torch.tensor([0.0, 255.0 * edge_opacity, 0.0], device=self.device, dtype=torch.float32)

            # Generate new particles at random edge points with modulated count
            num_new_particles = max(0, mod_num_particles - len(self.particles))  # Ensure non-negative count
            new_particles = self.create_particles(edges, angle, num_new_particles, particle_lifespan)

            # Add the new particles to the particle list
            self.particles.extend(new_particles)

            # Create particle layer
            particle_layer = torch.zeros_like(edge_layer)

            # Update particles and draw them onto particle_layer
            updated_particles = []
            for particle in self.particles:
                x, y, dx, dy, lifespan = particle

                # Move particle and clamp within bounds with modulated speed
                x = max(0, min(depth_normalized.shape[1] - 1, x + dx * mod_particle_speed))
                y = max(0, min(depth_normalized.shape[0] - 1, y + dy * mod_particle_speed))
                lifespan -= 1  # Decrease lifespan

                # Check boundaries and add to the updated list if within bounds and still alive
                if 0 <= x < depth_normalized.shape[1] and 0 <= y < depth_normalized.shape[0] and lifespan > 0:
                    updated_particles.append((x, y, dx, dy, lifespan))

                    # Draw particle as a small circle on the GPU within bounds
                    x_int, y_int = int(x), int(y)
                    x_end = min(x_int + particle_size, particle_layer.shape[1])  # Ensure x-end is within bounds
                    y_end = min(y_int + particle_size, particle_layer.shape[0])  # Ensure y-end is within bounds

                    particle_layer[y_int:y_end, x_int:x_end] = torch.tensor([0.0, 255.0, 0.0], device=self.device, dtype=torch.float32)

            # Update particles list
            self.particles = updated_particles

            # Combine edge and particle layers with adjustable opacity (modulated)
            combined_output = (edge_layer * edge_opacity + particle_layer * mod_particle_opacity).clamp(0, 255).byte()

            # Convert processed image back to tensor format and move to CPU for output
            output_image = Image.fromarray(combined_output.cpu().numpy())
            output_tensor = self.p2t(output_image)

            # Add to processed tensors list
            processed_tensors.append(output_tensor)

            # Update progress bar
            pbar.update_absolute(idx + 1)

        # Concatenate all processed tensors along batch dimension
        final_output = torch.cat(processed_tensors, dim=0)

        return (final_output,)

    def create_particles(self, edges, angle, num_new_particles, lifespan):
        """Create new particles at edge points with initial directions and lifespan."""
        particles = []
        edge_points = np.argwhere(edges > 0)

        # Ensure we don't sample more than available edge points
        if len(edge_points) > 0:
            sampled_points = random.sample(list(edge_points), min(len(edge_points), num_new_particles))

            for (y, x) in sampled_points:
                # Get the angle for direction from gradient, making sure to extract a scalar
                angle_value = angle[y, x].item() if angle[y, x].numel() == 1 else angle[y, x].mean().item()
                dx = np.cos(np.deg2rad(angle_value))
                dy = np.sin(np.deg2rad(angle_value))
                particles.append((float(x), float(y), float(dx), float(dy), lifespan))

        return particles

    def reset(self):
        """Reset the previous frame memory and particles for a new sequence."""
        self.prev_frame = None
        self.particles = []  # Reset particles for each new sequence

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
    "LuminanceParticleNode": LuminanceParticleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminanceParticleNode": "Luminance Particles"
}
