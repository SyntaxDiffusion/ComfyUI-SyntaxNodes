"""ComfyUI registration for Syntax Nodes.

Each node module is imported independently so a single module failing
(missing optional dependency, import-time error) only disables that
module's nodes instead of unloading the entire pack.
"""

import importlib
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# (module_name, [(class_name, mapping_key, display_name), ...])
_NODE_SPECS = [
    ("jigsaw_puzzle_node", [("JigsawPuzzleNode", "JigsawPuzzleNode", "Jigsaw Puzzle Effect")]),
    ("low_poly_node", [("LowPolyNode", "LowPolyNode", "Low Poly Image Processor")]),
    ("region_boundary_node", [("RegionBoundaryNode", "RegionBoundaryNode", "Region Boundary Node")]),
    ("pointillism", [("PointillismNode", "PointillismNode", "Pointillism Effect")]),
    ("frequency_beat_sync", [("FrequencyBeatSyncNode", "FrequencyBeatSyncNode", "Beat Sync")]),
    ("ghosting_afterimage_node", [("GhostingNode", "GhostingNode", "Ghosting/Afterimage Effect")]),
    ("depth_to_lidar_effect_node", [("DepthToLidarEffectNode", "DepthToLidarEffectNode", "Depth to LIDAR Effect")]),
    ("LuminanceParticleNode", [("LuminanceParticleNode", "LuminanceParticleNode", "Luminance Particle Effect")]),
    ("edge_measurement_overlay_node", [("EdgeMeasurementOverlayNode", "EdgeMeasurementOverlayNode", "Edge Measurement Overlay")]),
    ("edge_tracing_node", [("EdgeTracingNode", "EdgeTracingNode", "Edge Tracing Animation")]),
    ("variable_line_width_effect_node", [("VariableLineWidthEffectNode", "VariableLineWidthEffectNode", "Variable Line Width Effect")]),
    ("cyberpunk_window_node", [("CyberpunkWindowNode", "CyberpunkWindowNode", "Cyberpunk Window Effect")]),
    ("cyberpunk_magnify_node", [("CyberpunkMagnifyNode", "CyberpunkMagnifyNode", "Cyberpunk Magnify Effect")]),
    ("rgb_streak_node", [("RGBStreakNode", "RGBStreakNode", "RGB Streak Effect")]),
    ("voxel_node", [("VoxelNode", "VoxelNode", "Voxel Block Effect")]),
    ("papercraftnode", [("PaperCraftNode", "PaperCraftNode", "Paper Craft Effect")]),
    ("frequency_beat_sync_advanced", [("FrequencyBeatSyncNode", "FrequencyBeatSyncNodeAdvanced", "Beat Sync (Advanced)")]),
    ("pixel_scatter_node", [("PixelScatterNode", "PixelScatterNode", "Pixel Scatter Effect")]),
    ("audio_reactive_template", [("AudioReactiveTemplateNode", "AudioReactiveTemplateNode", "Audio-Reactive Template")]),
    ("ml_sharp_node", [
        ("MLSharpNode", "MLSharpNode", "SHARP 3D Gaussian Splat"),
        ("MLSharpBatchNode", "MLSharpBatchNode", "SHARP 3D Gaussian Splat (Batch)"),
    ]),
    ("preview_3d_gs_node", [
        ("Preview3DGaussianSplat", "Preview3DGaussianSplat", "Preview 3D Gaussian Splat"),
        ("PreviewGaussianSplatVideo", "PreviewGaussianSplatVideo", "Preview Gaussian Splat Video"),
        ("LoadGaussianSplat", "LoadGaussianSplat", "Load Gaussian Splat"),
        ("SaveGaussianSplat", "SaveGaussianSplat", "Save Gaussian Splat"),
    ]),
    ("prompt_travel_sampler_node", [("SyntaxPromptTravelKSampler", "SyntaxPromptTravelKSampler", "Prompt Travel KSampler")]),
    ("sdcn_feedback_animation", [("SDCNFeedbackAnimation", "SDCNFeedbackAnimation", "SD-CN Feedback Animation")]),
    ("sdcn_feedback_animation_audio", [("SDCNFeedbackAnimationAudio", "SDCNFeedbackAnimationAudio", "SD-CN Feedback Animation (Audio Reactive)")]),
    ("syntax_feedback_sampler", [("SyntaxFeedbackSampler", "SyntaxFeedbackSampler", "Feedback Sampler (Prompt Scheduled)")]),
]

for _module_name, _classes in _NODE_SPECS:
    try:
        _module = importlib.import_module(f".{_module_name}", __name__)
    except Exception:
        print(f"[Syntax Nodes] Failed to import '{_module_name}' — its nodes are disabled:")
        traceback.print_exc()
        continue
    for _class_name, _key, _display in _classes:
        _cls = getattr(_module, _class_name, None)
        if _cls is None:
            print(f"[Syntax Nodes] '{_module_name}' loaded but has no class '{_class_name}' — skipping")
            continue
        NODE_CLASS_MAPPINGS[_key] = _cls
        NODE_DISPLAY_NAME_MAPPINGS[_key] = _display

WEB_DIRECTORY = "web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print(f"[Syntax Nodes] Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
