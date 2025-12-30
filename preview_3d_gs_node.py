"""
Preview 3D Gaussian Splat Node for ComfyUI
Provides 3D viewer and video preview capabilities for Gaussian Splat PLY files.
"""

import os
import urllib.parse
import folder_paths
import server
from aiohttp import web


def normalize_path(path):
    """Normalize path separators for cross-platform compatibility."""
    return path.replace('\\', '/')


# Register route to serve PLY files
@server.PromptServer.instance.routes.get("/syntaxnodes/gsplat/ply")
async def serve_ply_file(request):
    """Serve a PLY file for the 3D viewer."""
    ply_path = request.query.get("path", "")
    if not ply_path:
        return web.Response(status=400, text="No path specified")

    # URL decode the path
    ply_path = urllib.parse.unquote(ply_path)

    # Security check - only serve .ply files
    if not ply_path.lower().endswith('.ply'):
        return web.Response(status=403, text="Only PLY files allowed")

    if not os.path.exists(ply_path):
        return web.Response(status=404, text="File not found")

    # Read and serve the file
    try:
        with open(ply_path, 'rb') as f:
            data = f.read()
        return web.Response(
            body=data,
            content_type='application/octet-stream',
            headers={
                'Content-Disposition': f'inline; filename="{os.path.basename(ply_path)}"',
                'Access-Control-Allow-Origin': '*',
            }
        )
    except Exception as e:
        return web.Response(status=500, text=str(e))


# Register route to serve the viewer HTML
@server.PromptServer.instance.routes.get("/syntaxnodes/gsplat/viewer")
async def serve_viewer(request):
    """Serve the Gaussian Splat viewer HTML."""
    viewer_path = os.path.join(os.path.dirname(__file__), "web", "html", "gsplat_viewer.html")
    if os.path.exists(viewer_path):
        with open(viewer_path, 'r', encoding='utf-8') as f:
            html = f.read()
        return web.Response(text=html, content_type='text/html')
    return web.Response(status=404, text="Viewer not found")


class Preview3DGaussianSplat:
    """
    Preview node for 3D Gaussian Splat PLY files.
    Displays an interactive 3D viewer using WebGL.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the .ply Gaussian Splat file"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = "SyntaxNodes/3D"
    OUTPUT_NODE = True

    def preview(self, ply_path, unique_id=None):
        """
        Return UI data for 3D preview.
        """
        if not ply_path or not os.path.exists(ply_path):
            return {"ui": {"gsplat": []}}

        # Normalize and encode the path for URL
        ply_path = normalize_path(ply_path)
        encoded_path = urllib.parse.quote(ply_path, safe='')

        # Return the viewer URL with PLY path
        viewer_url = f"/syntaxnodes/gsplat/viewer?ply=/syntaxnodes/gsplat/ply?path={encoded_path}"

        return {
            "ui": {
                "gsplat": [{
                    "url": viewer_url,
                    "ply_path": ply_path,
                }]
            }
        }


class PreviewGaussianSplatVideo:
    """
    Preview node for Gaussian Splat rendered videos.
    Displays the flythrough video from SHARP.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the rendered video file (.mp4)"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = "SyntaxNodes/3D"
    OUTPUT_NODE = True

    def preview(self, video_path, unique_id=None):
        """
        Return UI data for video preview.
        Uses the same format as ComfyUI's PreviewVideo.
        """
        if not video_path or not os.path.exists(video_path):
            return {"ui": {"images": [], "animated": (True,)}}

        output_dir = folder_paths.get_output_directory()

        # Normalize path for comparison
        video_path_normalized = normalize_path(video_path)
        output_dir_normalized = normalize_path(output_dir)

        if video_path_normalized.startswith(output_dir_normalized):
            rel_path = os.path.relpath(video_path, output_dir)
            subfolder = os.path.dirname(rel_path).replace('\\', '/')
            filename = os.path.basename(rel_path)
            folder_type = "output"
        else:
            subfolder = ""
            filename = os.path.basename(video_path)
            folder_type = "output"

        # Format matches ComfyUI's PreviewVideo: {"images": [...], "animated": (True,)}
        return {
            "ui": {
                "images": [{
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": folder_type,
                }],
                "animated": (True,)
            }
        }


class LoadGaussianSplat:
    """
    Load a Gaussian Splat PLY file from the input/3d or output directory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_3d_dir = os.path.join(folder_paths.get_input_directory(), "3d")
        os.makedirs(input_3d_dir, exist_ok=True)

        output_dir = folder_paths.get_output_directory()

        files = []

        if os.path.exists(input_3d_dir):
            for f in os.listdir(input_3d_dir):
                if f.lower().endswith('.ply'):
                    files.append(f"3d/{f}")

        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.lower().endswith('.ply'):
                    files.append(f"output:{f}")

        if not files:
            files = ["none"]

        return {
            "required": {
                "ply_file": (sorted(files),),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "load"
    CATEGORY = "SyntaxNodes/3D"

    @classmethod
    def IS_CHANGED(cls, ply_file):
        return float("nan")

    def load(self, ply_file):
        if ply_file == "none":
            return ("",)

        if ply_file.startswith("output:"):
            filename = ply_file[7:]
            full_path = os.path.join(folder_paths.get_output_directory(), filename)
        else:
            full_path = os.path.join(folder_paths.get_input_directory(), ply_file)

        if not os.path.exists(full_path):
            print(f"[LoadGaussianSplat] File not found: {full_path}")
            return ("",)

        return (normalize_path(full_path),)


class SaveGaussianSplat:
    """
    Save/copy a Gaussian Splat PLY file with a custom name.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the source PLY file"
                }),
                "output_name": ("STRING", {
                    "default": "gaussian_splat",
                    "tooltip": "Output filename (without extension)"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "SyntaxNodes/3D"
    OUTPUT_NODE = True

    def save(self, ply_path, output_name):
        import shutil

        if not ply_path or not os.path.exists(ply_path):
            print(f"[SaveGaussianSplat] Source file not found: {ply_path}")
            return ("",)

        output_dir = folder_paths.get_output_directory()

        counter = 0
        output_path = os.path.join(output_dir, f"{output_name}.ply")
        while os.path.exists(output_path):
            counter += 1
            output_path = os.path.join(output_dir, f"{output_name}_{counter:04d}.ply")

        shutil.copy2(ply_path, output_path)
        print(f"[SaveGaussianSplat] Saved to: {output_path}")

        return (normalize_path(output_path),)


NODE_CLASS_MAPPINGS = {
    "Preview3DGaussianSplat": Preview3DGaussianSplat,
    "PreviewGaussianSplatVideo": PreviewGaussianSplatVideo,
    "LoadGaussianSplat": LoadGaussianSplat,
    "SaveGaussianSplat": SaveGaussianSplat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Preview3DGaussianSplat": "Preview 3D Gaussian Splat",
    "PreviewGaussianSplatVideo": "Preview Gaussian Splat Video",
    "LoadGaussianSplat": "Load Gaussian Splat",
    "SaveGaussianSplat": "Save Gaussian Splat",
}
