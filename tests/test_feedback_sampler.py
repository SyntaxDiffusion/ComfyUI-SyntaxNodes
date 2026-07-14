"""Focused regression tests for Feedback Sampler (Prompt Scheduled)."""

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "_syntax_nodes_feedback_tests"


def _load_sampler_module():
    """Load the node without importing every node in the pack."""
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(ROOT)]
    sys.modules.setdefault(PACKAGE_NAME, package)

    if "comfy" not in sys.modules:
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        samplers = types.ModuleType("comfy.samplers")

        class KSampler:
            SAMPLERS = ("euler",)
            SCHEDULERS = ("normal",)

        samplers.KSampler = KSampler
        comfy.samplers = samplers
        sys.modules["comfy"] = comfy
        sys.modules["comfy.samplers"] = samplers

    sys.modules.setdefault("nodes", types.ModuleType("nodes"))

    module_name = f"{PACKAGE_NAME}.syntax_feedback_sampler"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "syntax_feedback_sampler.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SAMPLER_MODULE = _load_sampler_module()
SyntaxFeedbackSampler = SAMPLER_MODULE.SyntaxFeedbackSampler


class RecordingVAE:
    def __init__(self, encoded):
        self.encoded = encoded
        self.received = None

    def encode(self, pixels):
        self.received = pixels
        return self.encoded


def test_wan_temporal_pixels_are_not_given_an_extra_batch_dimension():
    sampler = SyntaxFeedbackSampler()
    expected = torch.zeros(1, 16, 1, 8, 8)
    vae = RecordingVAE(expected)
    temporal_pixels = np.zeros((1, 64, 64, 3), dtype=np.uint8)

    actual = sampler.image_to_latent(temporal_pixels, vae, expected.shape)

    assert vae.received.shape == (1, 64, 64, 3)
    assert actual is expected


def test_normal_image_pixels_still_gain_one_batch_dimension():
    sampler = SyntaxFeedbackSampler()
    expected = torch.zeros(1, 4, 8, 8)
    vae = RecordingVAE(expected)

    sampler.image_to_latent(np.zeros((64, 64, 3), dtype=np.uint8), vae, expected.shape)

    assert vae.received.shape == (1, 64, 64, 3)


def test_temporal_color_matching_is_framewise():
    sampler = SyntaxFeedbackSampler()
    source = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    reference = np.flip(source, axis=1).copy()

    actual = sampler.match_color_histogram(source, reference, "RGB")
    expected = np.stack([
        sampler.match_color_histogram(source[i], reference[i], "RGB")
        for i in range(source.shape[0])
    ])

    assert actual.shape == source.shape
    assert np.array_equal(actual, expected)


def test_sharpening_does_not_mix_color_or_temporal_axes():
    sampler = SyntaxFeedbackSampler()
    image = np.zeros((2, 9, 9, 3), dtype=np.uint8)
    image[..., 0] = 20
    image[..., 1] = 100
    image[..., 2] = 220

    sharpened = sampler.apply_sharpening(image, 0.5)

    assert np.array_equal(sharpened, image)


def test_zoom_preserves_5d_channel_time_order():
    sampler = SyntaxFeedbackSampler()
    latent = torch.empty(1, 2, 3, 8, 8)
    for channel in range(2):
        for frame in range(3):
            latent[:, channel, frame].fill_(channel * 10 + frame)

    zoomed, _ = sampler.zoom_latent(latent, 0.1)

    assert torch.allclose(zoomed, latent)


def test_zoom_out_uses_stable_reflection_instead_of_random_latents():
    sampler = SyntaxFeedbackSampler()
    latent = torch.empty(1, 2, 16, 16)
    latent[:, 0].fill_(2.0)
    latent[:, 1].fill_(-3.0)

    zoomed, automatic_mask = sampler.zoom_latent(latent, -0.25)

    assert torch.allclose(zoomed, latent)
    assert automatic_mask is None


def test_tiny_zoom_out_remains_centered():
    sampler = SyntaxFeedbackSampler()
    axis = torch.linspace(-1.0, 1.0, 128)
    symmetric = 1.0 - axis.abs()
    latent = symmetric[None, None, None, :].expand(1, 4, 128, 128).clone()

    zoomed, automatic_mask = sampler.zoom_latent(latent, -0.005)

    assert torch.allclose(zoomed, torch.flip(zoomed, dims=(-1,)), atol=1e-5)
    assert automatic_mask is None


def test_zoom_out_has_no_automatic_outpaint_path():
    source = (ROOT / "syntax_feedback_sampler.py").read_text(encoding="utf-8")

    assert "create_outpaint_mask" not in source
    assert "OUTPAINT_MIN_DENOISE" not in source
    assert source.count("denoise=feedback_denoise") >= 2


def test_unbatch_conditioning_preserves_and_slices_tensor_metadata():
    sampler = SyntaxFeedbackSampler()
    cond = torch.arange(3 * 4 * 2).reshape(3, 4, 2)
    pooled = torch.arange(3 * 5).reshape(3, 5)
    attention_mask = torch.arange(3 * 4).reshape(3, 4)
    shared = torch.ones(1, 7)
    conditioning = [[cond, {
        "pooled_output": pooled,
        "attention_mask": attention_mask,
        "shared": shared,
        "tag": "keep-me",
    }]]

    frames = sampler.unbatch_conditioning(conditioning)

    assert len(frames) == 3
    for i, frame in enumerate(frames):
        tensor, metadata = frame[0]
        assert torch.equal(tensor, cond[i:i + 1])
        assert torch.equal(metadata["pooled_output"], pooled[i:i + 1])
        assert torch.equal(metadata["attention_mask"], attention_mask[i:i + 1])
        assert metadata["shared"] is shared
        assert metadata["tag"] == "keep-me"


def test_only_consolidated_general_feedback_sampler_is_registered():
    source = (ROOT / "__init__.py").read_text(encoding="utf-8")

    assert '"SyntaxFeedbackSampler": SyntaxFeedbackSampler' in source
    assert '"FeedbackSampler":' not in source
    assert '"SDCNFeedbackAnimation": SDCNFeedbackAnimation' in source
    assert '"SDCNFeedbackAnimationAudio": SDCNFeedbackAnimationAudio' in source


def test_feedback_defaults_do_not_accumulate_enhancement_noise():
    required = SyntaxFeedbackSampler.INPUT_TYPES()["required"]
    optional = SyntaxFeedbackSampler.INPUT_TYPES()["optional"]

    assert required["seed_variation"][1]["default"] == "fixed"
    assert required["color_coherence"][1]["default"] == "None"
    assert required["noise_amount"][1]["default"] == 0.0
    assert required["sharpen_amount"][1]["default"] == 0.0
    assert required["contrast_boost"][1]["default"] == 1.0
    assert optional["frame_cadence"][1]["default"] == 1
    assert list(optional)[-1] == "frame_cadence"


def test_frame_cadence_keeps_frame_zero_and_interval_keyframes():
    sampled = [
        i for i in range(8)
        if SyntaxFeedbackSampler.should_diffuse_frame(i, 3)
    ]

    assert sampled == [0, 3, 6]
    assert all(
        SyntaxFeedbackSampler.should_diffuse_frame(i, 1)
        for i in range(8)
    )


def test_frame_cadence_skips_sampler_but_keeps_every_output_frame():
    class RecordingSampler(SyntaxFeedbackSampler):
        def __init__(self):
            self.sampled_seeds = []

        def sample_with_callback(self, model, seed, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent, denoise,
                                 callback):
            self.sampled_seeds.append(seed)
            return {"samples": latent["samples"] + 1}

    class QuietProgress:
        def __init__(self, *args, **kwargs):
            pass

        def begin_frame(self, *args, **kwargs):
            pass

        def sampling_callback(self, *args, **kwargs):
            return lambda *callback_args: None

        def complete_frame(self, *args, **kwargs):
            pass

        def finish(self):
            pass

    sampler = RecordingSampler()
    conditioning = [[torch.zeros(1, 1, 1), {}]]
    original_progress = SAMPLER_MODULE.SyntaxFeedbackProgress
    SAMPLER_MODULE.SyntaxFeedbackProgress = QuietProgress
    try:
        final, all_frames = sampler.sample(
            model=object(), seed=10, steps=1, cfg=1.0,
            sampler_name="euler", scheduler="normal",
            latent_image={"samples": torch.zeros(1, 4, 8, 8)},
            denoise=1.0, zoom_value=0.0, iterations=5,
            feedback_denoise=0.3, seed_variation="increment",
            angle=0.0, translation_x=0.0, translation_y=0.0,
            translation_z=0.0, rotation_3d_x=0.0,
            rotation_3d_y=0.0, rotation_3d_z=0.0,
            color_coherence="None", noise_amount=0.0,
            noise_type="gaussian", sharpen_amount=0.0,
            contrast_boost=1.0, frame_cadence=2,
            positive=conditioning, negative=conditioning,
        )
    finally:
        SAMPLER_MODULE.SyntaxFeedbackProgress = original_progress

    assert sampler.sampled_seeds == [10, 12, 14]
    assert all_frames["samples"].shape[0] == 5
    assert torch.equal(
        all_frames["samples"][:, 0, 0, 0],
        torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0]),
    )
    assert torch.equal(final["samples"], all_frames["samples"][-1:])


def test_krea2_detection_is_narrow_and_does_not_change_other_models():
    Krea2Tokenizer = type("Krea2Tokenizer", (), {})
    FluxTokenizer = type("FluxTokenizer", (), {})
    krea_clip = types.SimpleNamespace(tokenizer=Krea2Tokenizer())
    flux_clip = types.SimpleNamespace(tokenizer=FluxTokenizer())

    assert SyntaxFeedbackSampler.is_krea2_pipeline(clip=krea_clip)
    assert not SyntaxFeedbackSampler.is_krea2_pipeline(clip=flux_clip)
    assert not SyntaxFeedbackSampler.is_krea2_pipeline()


def test_krea_transition_average_is_limited_to_structural_steps():
    split = SyntaxFeedbackSampler.krea_transition_timestep_split(0.3)

    assert abs(split - 0.775) < 1e-9
    assert SyntaxFeedbackSampler.krea_transition_timestep_split(1.0) == 0.25
    assert SyntaxFeedbackSampler.krea_transition_timestep_split(0.0) == 1.0


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} feedback sampler tests passed.")
