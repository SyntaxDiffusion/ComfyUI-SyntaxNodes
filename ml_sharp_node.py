import os
import ssl
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import folder_paths

# Model URL for automatic download
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
MODEL_FILENAME = "sharp_2572gikvuh.pt"


class MLSharpNode:
    """
    ComfyUI node for SHARP (Single-image 3D Gaussian Splat Generation).
    Takes an input image and generates a 3D Gaussian Splat PLY file for novel view synthesis.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None
        self.sharp_available = False
        self._check_sharp_installation()

    def _check_sharp_installation(self):
        """Check if sharp is installed and available."""
        try:
            from sharp.models import create_predictor, PredictorParams
            from sharp.utils.gaussians import unproject_gaussians, save_ply
            self.sharp_available = True
        except ImportError as e:
            self.sharp_available = False
            print(f"[MLSharpNode] Warning: 'sharp' package not found: {e}")
            print("[MLSharpNode] Install with: pip install -e . from ml-sharp repo")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "focal_length_mm": ("FLOAT", {
                    "default": 30.0,
                    "min": 10.0,
                    "max": 200.0,
                    "step": 1.0,
                    "tooltip": "Focal length in mm (35mm equivalent). Default 30mm if unknown."
                }),
                "output_filename": ("STRING", {
                    "default": "sharp_output",
                    "tooltip": "Base filename for the output PLY file (without extension)"
                }),
                "render_video": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Render a flythrough video of the 3D scene (requires CUDA)"
                }),
                "video_frames": ("INT", {
                    "default": 60,
                    "min": 10,
                    "max": 300,
                    "step": 10,
                    "tooltip": "Number of frames for the flythrough video (more = longer render time)"
                }),
                "camera_path": (["rotate_forward", "rotate", "swipe", "shake"], {
                    "default": "rotate_forward",
                    "tooltip": "Camera trajectory type: rotate_forward (orbit + push in), rotate (orbit), swipe (side-to-side), shake (small motion)"
                }),
                "camera_distance": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Camera distance from subject (0 = auto based on scene)"
                }),
                "max_disparity": ("FLOAT", {
                    "default": 0.08,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Maximum parallax/3D effect strength (higher = more dramatic camera movement)"
                }),
                "max_zoom": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "tooltip": "Maximum zoom amount for rotate_forward path"
                }),
                "loop_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Number of times to repeat the camera path"
                }),
            },
            "optional": {
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional path to custom model checkpoint (.pt file). Leave empty for default."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "EXTRINSICS", "INTRINSICS")
    RETURN_NAMES = ("ply_path", "video_path", "preview_image", "extrinsics", "intrinsics")
    FUNCTION = "process_image"
    CATEGORY = "SyntaxNodes/3D"
    OUTPUT_NODE = True

    def _get_model_path(self):
        """Get the path where the model should be stored."""
        # Use ComfyUI's models directory structure
        models_dir = folder_paths.models_dir
        sharp_dir = os.path.join(models_dir, "sharp")
        os.makedirs(sharp_dir, exist_ok=True)
        return os.path.join(sharp_dir, MODEL_FILENAME)

    def _download_model_with_ssl_fallback(self, url, dest_path):
        """Download model with SSL certificate verification fallback."""
        import urllib.request
        import shutil

        print(f"[MLSharpNode] Downloading model to: {dest_path}")

        # First try with normal SSL verification
        try:
            print("[MLSharpNode] Attempting download with SSL verification...")
            with urllib.request.urlopen(url) as response:
                with open(dest_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            print("[MLSharpNode] Download complete!")
            return True
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                print("[MLSharpNode] SSL verification failed, trying with unverified context...")
                # Create unverified SSL context as fallback
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                try:
                    with urllib.request.urlopen(url, context=ssl_context) as response:
                        with open(dest_path, 'wb') as out_file:
                            shutil.copyfileobj(response, out_file)
                    print("[MLSharpNode] Download complete (with SSL bypass)!")
                    return True
                except Exception as e2:
                    print(f"[MLSharpNode] Download failed even with SSL bypass: {e2}")
                    return False
            else:
                print(f"[MLSharpNode] Download failed: {e}")
                return False
        except Exception as e:
            print(f"[MLSharpNode] Download failed: {e}")
            return False

    def _load_model(self, checkpoint_path=""):
        """Load the SHARP model."""
        if not self.sharp_available:
            raise RuntimeError("sharp package is not installed. Please install ml-sharp first.")

        from sharp.models import create_predictor, PredictorParams

        if self.predictor is not None:
            return  # Model already loaded

        print("[MLSharpNode] Loading SHARP model...")

        # Determine which checkpoint to use
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[MLSharpNode] Loading from custom checkpoint: {checkpoint_path}")
            model_path = checkpoint_path
        else:
            # Check if model exists in ComfyUI models folder
            model_path = self._get_model_path()

            # Also check torch hub cache
            torch_cache = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", MODEL_FILENAME)

            if os.path.exists(model_path):
                print(f"[MLSharpNode] Loading from: {model_path}")
            elif os.path.exists(torch_cache):
                print(f"[MLSharpNode] Loading from torch cache: {torch_cache}")
                model_path = torch_cache
            else:
                # Need to download
                print(f"[MLSharpNode] Model not found, downloading from Apple CDN...")
                if not self._download_model_with_ssl_fallback(DEFAULT_MODEL_URL, model_path):
                    raise RuntimeError(
                        f"Failed to download model. Please download manually from:\n"
                        f"  {DEFAULT_MODEL_URL}\n"
                        f"And place it at:\n"
                        f"  {model_path}\n"
                        f"Or specify the path in the 'checkpoint_path' input."
                    )

        # Load state dict
        state_dict = torch.load(model_path, weights_only=True, map_location=self.device)

        # Create and load predictor
        self.predictor = create_predictor(PredictorParams())
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()
        self.predictor.to(self.device)

        print("[MLSharpNode] Model loaded successfully!")

    def _convert_focal_length(self, width: float, height: float, f_mm: float = 30) -> float:
        """Converts a focal length given in mm to pixels (35mm film equivalent)."""
        return f_mm * np.sqrt(width**2.0 + height**2.0) / np.sqrt(36**2 + 24**2)

    def _build_extrinsics(self) -> list:
        """Build 4x4 identity extrinsics matrix (camera-to-world).

        Returns identity matrix since SHARP predictions are in camera-centric space.
        """
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]

    def _build_intrinsics(self, f_px: float, width: int, height: int) -> list:
        """Build 3x3 camera intrinsics matrix.

        Args:
            f_px: Focal length in pixels
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            3x3 intrinsics matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        """
        cx = width / 2.0
        cy = height / 2.0
        return [
            [f_px, 0.0, cx],
            [0.0, f_px, cy],
            [0.0, 0.0, 1.0]
        ]

    @torch.no_grad()
    def _predict_image(self, image_np: np.ndarray, f_px: float) -> "Gaussians3D":
        """Predict Gaussians from an image (mirrors sharp.cli.predict.predict_image)."""
        import torch.nn.functional as F
        from sharp.utils.gaussians import unproject_gaussians

        internal_shape = (1536, 1536)

        # Convert to tensor
        image_pt = torch.from_numpy(image_np.copy()).float().to(self.device).permute(2, 0, 1) / 255.0
        _, height, width = image_pt.shape
        disparity_factor = torch.tensor([f_px / width]).float().to(self.device)

        # Resize to internal resolution
        image_resized_pt = F.interpolate(
            image_pt[None],
            size=(internal_shape[1], internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )

        print(f"[MLSharpNode] Running inference...")

        # Predict Gaussians in NDC space
        gaussians_ndc = self.predictor(image_resized_pt, disparity_factor)

        print(f"[MLSharpNode] Running postprocessing...")

        # Build intrinsics matrix
        intrinsics = torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).float().to(self.device)

        # Scale intrinsics for resized image
        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= internal_shape[0] / width
        intrinsics_resized[1] *= internal_shape[1] / height

        # Convert Gaussians to metric space
        gaussians = unproject_gaussians(
            gaussians_ndc,
            torch.eye(4).to(self.device),
            intrinsics_resized,
            internal_shape
        )

        return gaussians

    def _render_video(self, gaussians, f_px, width, height, video_path, num_frames=60,
                       camera_path="rotate_forward", camera_distance=0.0, max_disparity=0.08,
                       max_zoom=0.15, loop_count=1):
        """Render a flythrough video of the gaussians with custom camera path."""
        try:
            from sharp.utils import camera, gsplat, io as sharp_io
            from sharp.utils.gaussians import SceneMetaData
            from comfy.utils import ProgressBar

            if not torch.cuda.is_available():
                print("[MLSharpNode] Video rendering requires CUDA, skipping...")
                return False

            total_frames = num_frames * loop_count
            print(f"[MLSharpNode] Rendering {total_frames}-frame flythrough video on CUDA...")
            print(f"[MLSharpNode] Camera path: {camera_path}, disparity: {max_disparity}, zoom: {max_zoom}")

            device = torch.device("cuda")
            metadata = SceneMetaData(f_px, (width, height), "linearRGB")

            intrinsics = torch.tensor(
                [
                    [f_px, 0, (width - 1) / 2.0, 0],
                    [0, f_px, (height - 1) / 2.0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                device=device,
                dtype=torch.float32,
            )

            # Configure trajectory with custom parameters
            params = camera.TrajectoryParams()
            params.type = camera_path
            params.num_steps = num_frames
            params.num_repeats = loop_count
            params.max_disparity = max_disparity
            params.max_zoom = max_zoom
            if camera_distance > 0:
                params.distance_m = camera_distance

            camera_model = camera.create_camera_model(
                gaussians, intrinsics, resolution_px=metadata.resolution_px
            )
            trajectory = camera.create_eye_trajectory(
                gaussians, params, resolution_px=metadata.resolution_px, f_px=f_px
            )

            renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
            video_writer = sharp_io.VideoWriter(Path(video_path))

            # Move gaussians to GPU once, outside the loop
            gaussians_gpu = gaussians.to(device)

            # Progress bar for ComfyUI (trajectory length = num_steps * num_repeats)
            pbar = ProgressBar(total_frames)

            for frame_idx, eye_position in enumerate(trajectory):
                camera_info = camera_model.compute(eye_position)
                rendering_output = renderer(
                    gaussians_gpu,
                    extrinsics=camera_info.extrinsics[None].to(device),
                    intrinsics=camera_info.intrinsics[None].to(device),
                    image_width=camera_info.width,
                    image_height=camera_info.height,
                )
                color = (rendering_output.color[0].permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
                depth = rendering_output.depth[0]
                video_writer.add_frame(color, depth)
                pbar.update_absolute(frame_idx + 1)

            video_writer.close()
            print(f"[MLSharpNode] Saved video to: {video_path}")
            return True

        except Exception as e:
            print(f"[MLSharpNode] Video rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_image(self, image, focal_length_mm=30.0, output_filename="sharp_output",
                      render_video=True, video_frames=60, camera_path="rotate_forward",
                      camera_distance=0.0, max_disparity=0.08, max_zoom=0.15,
                      loop_count=1, checkpoint_path=""):
        """
        Process an image through SHARP to generate a 3D Gaussian Splat.

        Args:
            image: ComfyUI image tensor (B, H, W, C)
            focal_length_mm: Focal length in mm (35mm equivalent)
            output_filename: Base filename for output PLY
            render_video: Whether to render a flythrough video
            video_frames: Number of frames for the video
            camera_path: Camera trajectory type (rotate_forward, rotate, swipe, shake)
            camera_distance: Distance from subject (0 = auto)
            max_disparity: Maximum parallax/3D effect strength
            max_zoom: Maximum zoom amount
            loop_count: Number of times to repeat the camera path
            checkpoint_path: Optional path to custom model checkpoint

        Returns:
            tuple: (ply_file_path, video_path, preview_image, extrinsics, intrinsics)
                - extrinsics: 4x4 camera-to-world matrix (identity, list[list[float]]) or None on error
                - intrinsics: 3x3 camera intrinsics matrix (list[list[float]]) or None on error
        """
        if not self.sharp_available:
            # Return error message if sharp not installed
            error_msg = "ERROR: sharp package not installed. Install ml-sharp first."
            print(f"[MLSharpNode] {error_msg}")
            # Return original image as preview with empty paths
            return ("", "", image, None, None)

        from sharp.utils.gaussians import save_ply

        # Load model if not already loaded
        self._load_model(checkpoint_path)

        # Process first image in batch
        single_image = image[0]  # Take first image from batch

        # Convert ComfyUI tensor to numpy array (H, W, C) with values 0-255
        img_array = (single_image.cpu().numpy() * 255).astype(np.uint8)
        height, width = img_array.shape[:2]

        # Ensure RGB (3 channels)
        if img_array.ndim < 3 or img_array.shape[2] == 1:
            img_array = np.dstack((img_array, img_array, img_array))
        elif img_array.shape[2] > 3:
            img_array = img_array[:, :, :3]

        # Calculate focal length in pixels
        f_px = self._convert_focal_length(width, height, focal_length_mm)

        print(f"[MLSharpNode] Processing image {width}x{height}, focal_length={f_px:.1f}px (from {focal_length_mm}mm)")

        # Run inference
        gaussians = self._predict_image(img_array, f_px)

        # Prepare output path
        output_dir = folder_paths.get_output_directory()

        # Add unique suffix to prevent overwrites
        counter = 0
        base_path = os.path.join(output_dir, f"{output_filename}.ply")
        ply_path = base_path
        while os.path.exists(ply_path):
            counter += 1
            ply_path = os.path.join(output_dir, f"{output_filename}_{counter:04d}.ply")

        # Save PLY file
        save_ply(gaussians, f_px, (height, width), Path(ply_path))
        print(f"[MLSharpNode] Saved PLY to: {ply_path}")

        # Optionally render video
        video_path = ""
        if render_video:
            video_path = ply_path.replace(".ply", ".mp4")
            if not self._render_video(
                gaussians, f_px, width, height, video_path,
                num_frames=video_frames,
                camera_path=camera_path,
                camera_distance=camera_distance,
                max_disparity=max_disparity,
                max_zoom=max_zoom,
                loop_count=loop_count
            ):
                video_path = ""

        # Build camera matrices for output
        extrinsics = self._build_extrinsics()
        intrinsics = self._build_intrinsics(f_px, width, height)
        return (ply_path, video_path, image, extrinsics, intrinsics)


class MLSharpBatchNode:
    """
    Batch processing version of MLSharp - processes multiple images at once.
    """

    def __init__(self):
        self.sharp_node = MLSharpNode()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "focal_length_mm": ("FLOAT", {
                    "default": 30.0,
                    "min": 10.0,
                    "max": 200.0,
                    "step": 1.0,
                }),
                "output_prefix": ("STRING", {
                    "default": "sharp_batch",
                }),
            },
            "optional": {
                "checkpoint_path": ("STRING", {
                    "default": "",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_paths",)
    FUNCTION = "process_batch"
    CATEGORY = "SyntaxNodes/3D"
    OUTPUT_NODE = True

    def process_batch(self, images, focal_length_mm=30.0, output_prefix="sharp_batch",
                      checkpoint_path=""):
        """Process a batch of images through SHARP."""
        from comfy.utils import ProgressBar

        batch_size = images.shape[0]
        pbar = ProgressBar(batch_size)

        ply_paths = []

        for idx in range(batch_size):
            single_image = images[idx:idx+1]
            filename = f"{output_prefix}_{idx:04d}"

            ply_path, _, _, _, _ = self.sharp_node.process_image(
                single_image,
                focal_length_mm,
                filename,
                False,  # Don't render video for batch (too slow)
                checkpoint_path
            )

            ply_paths.append(ply_path)
            pbar.update_absolute(idx + 1)

        # Return paths as newline-separated string
        return ("\n".join(ply_paths),)


NODE_CLASS_MAPPINGS = {
    "MLSharpNode": MLSharpNode,
    "MLSharpBatchNode": MLSharpBatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLSharpNode": "SHARP 3D Gaussian Splat",
    "MLSharpBatchNode": "SHARP 3D Gaussian Splat (Batch)",
}
