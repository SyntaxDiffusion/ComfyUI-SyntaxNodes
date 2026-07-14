"""ComfyUI registration for Syntax Nodes."""

from .jigsaw_puzzle_node import JigsawPuzzleNode
from .low_poly_node import LowPolyNode
from .region_boundary_node import RegionBoundaryNode
from .pointillism import PointillismNode
from .frequency_beat_sync import FrequencyBeatSyncNode
from .ghosting_afterimage_node import GhostingNode
from .depth_to_lidar_effect_node import DepthToLidarEffectNode
from .LuminanceParticleNode import LuminanceParticleNode
from .edge_measurement_overlay_node import EdgeMeasurementOverlayNode
from .edge_tracing_node import EdgeTracingNode
from .variable_line_width_effect_node import VariableLineWidthEffectNode
from .cyberpunk_window_node import CyberpunkWindowNode
from .cyberpunk_magnify_node import CyberpunkMagnifyNode
from .rgb_streak_node import RGBStreakNode
from .voxel_node import VoxelNode
from .papercraftnode import PaperCraftNode
from .frequency_beat_sync_advanced import FrequencyBeatSyncNode as FrequencyBeatSyncNodeAdvanced
from .pixel_scatter_node import PixelScatterNode
from .audio_reactive_template import AudioReactiveTemplateNode
from .ml_sharp_node import MLSharpNode, MLSharpBatchNode
from .preview_3d_gs_node import Preview3DGaussianSplat, PreviewGaussianSplatVideo, LoadGaussianSplat, SaveGaussianSplat
from .prompt_travel_sampler_node import SyntaxPromptTravelKSampler
from .sdcn_feedback_animation import SDCNFeedbackAnimation
from .sdcn_feedback_animation_audio import SDCNFeedbackAnimationAudio
from .syntax_feedback_sampler import SyntaxFeedbackSampler


NODE_CLASS_MAPPINGS = {
    "JigsawPuzzleNode": JigsawPuzzleNode,
    "LowPolyNode": LowPolyNode,
    "RegionBoundaryNode": RegionBoundaryNode,
    "PointillismNode": PointillismNode,
    "FrequencyBeatSyncNode": FrequencyBeatSyncNode,
    "GhostingNode": GhostingNode,
    "DepthToLidarEffectNode": DepthToLidarEffectNode,
    "LuminanceParticleNode": LuminanceParticleNode,
    "EdgeMeasurementOverlayNode": EdgeMeasurementOverlayNode,
    "EdgeTracingNode": EdgeTracingNode,
    "VariableLineWidthEffectNode": VariableLineWidthEffectNode,
    "CyberpunkWindowNode": CyberpunkWindowNode,
    "CyberpunkMagnifyNode": CyberpunkMagnifyNode,
    "RGBStreakNode": RGBStreakNode,
    "VoxelNode": VoxelNode,
    "PaperCraftNode": PaperCraftNode,
    "FrequencyBeatSyncNodeAdvanced": FrequencyBeatSyncNodeAdvanced,
    "PixelScatterNode": PixelScatterNode,
    "AudioReactiveTemplateNode": AudioReactiveTemplateNode,
    "MLSharpNode": MLSharpNode,
    "MLSharpBatchNode": MLSharpBatchNode,
    "Preview3DGaussianSplat": Preview3DGaussianSplat,
    "PreviewGaussianSplatVideo": PreviewGaussianSplatVideo,
    "LoadGaussianSplat": LoadGaussianSplat,
    "SaveGaussianSplat": SaveGaussianSplat,
    "SyntaxPromptTravelKSampler": SyntaxPromptTravelKSampler,
    "SDCNFeedbackAnimation": SDCNFeedbackAnimation,
    "SDCNFeedbackAnimationAudio": SDCNFeedbackAnimationAudio,
    "SyntaxFeedbackSampler": SyntaxFeedbackSampler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "JigsawPuzzleNode": "Jigsaw Puzzle Effect",
    "LowPolyNode": "Low Poly Image Processor",
    "RegionBoundaryNode": "Region Boundary Node",
    "PointillismNode": "Pointillism Effect",
    "FrequencyBeatSyncNode": "Beat Sync",
    "GhostingNode": "Ghosting/Afterimage Effect",
    "DepthToLidarEffectNode": "Depth to LIDAR Effect",
    "LuminanceParticleNode": "Luminance Particle Effect",
    "EdgeMeasurementOverlayNode": "Edge Measurement Overlay",
    "EdgeTracingNode": "Edge Tracing Animation",
    "VariableLineWidthEffectNode": "Variable Line Width Effect",
    "CyberpunkWindowNode": "Cyberpunk Window Effect",
    "CyberpunkMagnifyNode": "Cyberpunk Magnify Effect",
    "RGBStreakNode": "RGB Streak Effect",
    "VoxelNode": "Voxel Block Effect",
    "PaperCraftNode": "Paper Craft Effect",
    "FrequencyBeatSyncNodeAdvanced": "Beat Sync (Advanced)",
    "PixelScatterNode": "Pixel Scatter Effect",
    "AudioReactiveTemplateNode": "Audio-Reactive Template",
    "MLSharpNode": "SHARP 3D Gaussian Splat",
    "MLSharpBatchNode": "SHARP 3D Gaussian Splat (Batch)",
    "Preview3DGaussianSplat": "Preview 3D Gaussian Splat",
    "PreviewGaussianSplatVideo": "Preview Gaussian Splat Video",
    "LoadGaussianSplat": "Load Gaussian Splat",
    "SaveGaussianSplat": "Save Gaussian Splat",
    "SyntaxPromptTravelKSampler": "Prompt Travel KSampler",
    "SDCNFeedbackAnimation": "SD-CN Feedback Animation",
    "SDCNFeedbackAnimationAudio": "SD-CN Feedback Animation (Audio Reactive)",
    "SyntaxFeedbackSampler": "Feedback Sampler (Prompt Scheduled)",
}


WEB_DIRECTORY = "web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("[Syntax Nodes] Custom nodes loaded")
