import numpy as np
import cv2
from PIL import Image
import torch
from comfy.utils import ProgressBar
from .audio_envelope_handler import AudioEnvelopeHandler

class GhostingNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_buffer = []  # Buffer to store recent frames
        
    @classmethod
    def INPUT_TYPES(cls):
        # Get standard audio inputs
        audio_inputs = AudioEnvelopeHandler.get_standard_inputs()

        return {
            "required": {
                "image": ("IMAGE",),
                "decay_rate": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01
                }),
                "blend_opacity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "buffer_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1
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
    
    def process_image(self, image, decay_rate, blend_opacity, buffer_size, frame_index=0, mask=None,
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
            print(f"[GhostingNode] Mapping {batch_size} video frames to {envelope_total_frames} envelope frames (scale={frame_scale:.2f}x)")
        else:
            frame_scale = 1.0
            print(f"[GhostingNode] Using 1:1 frame mapping (no envelope or batch_size={batch_size})")

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
            # Low freq (kick) → buffer_size (more trails on kick)
            kick_val = stems['kick']
            buffer_add = kick_val * 10.0 * envelope_intensity
            mod_buffer_size = int(buffer_size + buffer_add)

            # Mid freq (snare) → blend_opacity (flash trails on snare)
            snare_val = stems['snare']
            opacity_multiplier = 1.0 + (snare_val * 1.5 * envelope_intensity)
            mod_blend_opacity = blend_opacity * opacity_multiplier

            # Bass → decay_rate (slower decay on bass)
            bass_val = stems['bass']
            decay_multiplier = 1.0 + (bass_val * 2.0 * envelope_intensity)
            mod_decay_rate = decay_rate * decay_multiplier

            # Ensure valid ranges
            mod_buffer_size = int(np.clip(mod_buffer_size, 1, 20))
            mod_blend_opacity = float(np.clip(mod_blend_opacity, 0.0, 1.0))
            mod_decay_rate = float(np.clip(mod_decay_rate, 0.0, 5.0))

            # DEBUG: Show modulation for first few frames or when there's activity
            has_activity = kick_val > 0.1 or snare_val > 0.1 or bass_val > 0.1
            if has_activity or idx < 3:
                print(f"[GhostingNode] Video frame {idx} → Envelope frame {envelope_frame}:")
                print(f"  STEMS: kick={kick_val:.3f}, snare={snare_val:.3f}, bass={bass_val:.3f}")
                print(f"  RESULT: buffer_size {buffer_size}→{mod_buffer_size}, opacity {blend_opacity:.2f}→{mod_blend_opacity:.2f}, decay {decay_rate:.2f}→{mod_decay_rate:.2f}")

            # Extract single image and mask from batch
            single_image = image[idx:idx+1]
            single_mask = None
            if mask is not None:
                if mask.shape[0] > 1:
                    single_mask = mask[idx:idx+1]
                else:
                    single_mask = mask

            # Process the single image with modulated parameters
            output, = self._process_single_image(single_image, mod_decay_rate, mod_blend_opacity, mod_buffer_size, single_mask)
            processed_images.append(output)
            pbar.update_absolute(idx + 1)

        # Concatenate results and return
        return (torch.cat(processed_images, dim=0),)
    
    def _process_single_image(self, image, decay_rate, blend_opacity, buffer_size, mask=None):
        # Convert from ComfyUI image format to numpy array
        pil_image = self.t2p(image)
        frame = np.array(pil_image, dtype=np.float32)
        
        # Handle optional mask input
        if mask is not None:
            mask_pil = self.t2p(mask)
            if mask_pil.mode != 'L':  # Convert to grayscale if not already
                mask_pil = mask_pil.convert('L')
            mask_np = np.array(mask_pil, dtype=np.float32) / 255.0  # Normalize mask to [0, 1]
        else:
            mask_np = np.ones_like(frame[..., 0], dtype=np.float32)  # Default to full mask
        
        if self.frame_buffer:
            buffer_shape = self.frame_buffer[0].shape
            if frame.shape != buffer_shape:
                frame = cv2.resize(frame, (buffer_shape[1], buffer_shape[0]))
                if frame.shape[-1] != buffer_shape[-1]:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) if buffer_shape[-1] == 3 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        if len(self.frame_buffer) >= buffer_size:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(frame)
        
        ghost_frame = frame.copy()
        for i, previous_frame in enumerate(reversed(self.frame_buffer)):
            weight = blend_opacity * (decay_rate ** i)
            blended = cv2.addWeighted(ghost_frame, 1 - weight, previous_frame, weight, 0)
            ghost_frame = ghost_frame * (1 - mask_np[..., None]) + blended * mask_np[..., None]  # Apply mask
        
        ghost_frame = np.clip(ghost_frame, 0, 255).astype(np.uint8)
        ghost_pil_image = Image.fromarray(ghost_frame)
        output_tensor = self.p2t(ghost_pil_image)
        return (output_tensor,)
    
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

class ImageListToImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE", ),
                  }
            }
    
    INPUT_IS_LIST = True
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "doit"
    
    CATEGORY = "SyntaxNodes/Conversion"
    
    def doit(self, images):
        if len(images) <= 1:
            return (images[0],)
        else:
            image1 = images[0]
            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "lanczos", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)
            return (image1,)


class ImageBatchToImageList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), }}
    
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"
    
    CATEGORY = "SyntaxNodes/Conversion"
    
    def doit(self, image):
        images = [image[i:i + 1, ...] for i in range(image.shape[0])]
        return (images, )


NODE_CLASS_MAPPINGS = {
    "GhostingNode": GhostingNode,
    "ImageListToImageBatch": ImageListToImageBatch,
    "ImageBatchToImageList": ImageBatchToImageList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GhostingNode": "Ghosting/Afterimage Effect",
    "ImageListToImageBatch": "Image List to Batch",
    "ImageBatchToImageList": "Image Batch to List"
}