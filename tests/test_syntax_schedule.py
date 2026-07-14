"""Tests for the syntax_schedule package (ported FizzNodes batch prompt scheduling).

Covers the variable-length encoder case (Qwen/Flux Krea) where different
prompts encode to different token counts. Run standalone:
    python tests/test_syntax_schedule.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syntax_schedule import schedule_conditioning
from syntax_schedule.ScheduleFuncs import process_input_text

DIM = 64
SHORT_LEN = 126
LONG_LEN = 145


class MockClip:
    """Encodes prompts containing 'long' to LONG_LEN tokens, else SHORT_LEN."""

    def __init__(self):
        self.encode_calls = 0

    def tokenize(self, text):
        return text

    def encode_from_tokens(self, tokens, return_pooled=True, return_dict=False):
        self.encode_calls += 1
        seq = LONG_LEN if "long" in tokens else SHORT_LEN
        if "photorealistic" in tokens:
            value = 7.0
        elif "painting" in tokens:
            value = 3.0
        else:
            value = 1.0
        cond = torch.full((1, seq, DIM), value)
        pooled = torch.full((1, DIM), value)
        attention_mask = torch.ones(1, seq, dtype=torch.long)
        if return_dict:
            return {
                "cond": cond,
                "pooled_output": pooled,
                "attention_mask": attention_mask,
                "encoder_name": "mock-native",
            }
        return cond, pooled


def test_varlen_schedule():
    frames = 12
    # Second keyframe encodes longer than the first - the case that crashed
    # BatchPromptSchedule with Krea/Qwen models before the padding fix.
    text = '"0" :"short cat --neg blurry", "8" :"a much longer dog prompt long"'

    clip = MockClip()
    pos, neg = schedule_conditioning(text, clip, frames)

    # Each unique prompt must be encoded once, not once per frame:
    # 2 unique positive + 2 unique negative prompts across 12 frames = 4 encodes
    assert clip.encode_calls <= 4, f"expected <=4 encodes (unique prompts), got {clip.encode_calls}"

    pos_tensor = pos[0][0]
    neg_tensor = neg[0][0]
    assert pos_tensor.shape == (frames, LONG_LEN, DIM), f"pos shape {pos_tensor.shape}"
    assert neg_tensor.shape[0] == frames, f"neg batch {neg_tensor.shape}"
    assert "pooled_output" in pos[0][1], "missing pooled_output"
    assert pos[0][1]["pooled_output"].shape[0] == frames
    assert pos[0][1]["attention_mask"].shape == (frames, LONG_LEN)
    # At the exact first keyframe, padding added for the later long prompt must
    # remain masked so it cannot alter the current prompt's meaning.
    assert pos[0][1]["attention_mask"][0].sum() == SHORT_LEN
    assert pos[0][1]["encoder_name"] == "mock-native"
    print(f"PASS varlen: pos {tuple(pos_tensor.shape)}, neg {tuple(neg_tensor.shape)}")


def test_uniform_schedule():
    frames = 6
    text = '"0" :"a cat", "3" :"a dog"'

    pos, neg = schedule_conditioning(text, MockClip(), frames)

    pos_tensor = pos[0][0]
    assert pos_tensor.shape == (frames, SHORT_LEN, DIM), f"pos shape {pos_tensor.shape}"
    print(f"PASS uniform: pos {tuple(pos_tensor.shape)}")


def test_plain_prompt_is_reused_for_the_whole_batch():
    frames = 10
    clip = MockClip()

    pos, neg = schedule_conditioning("photorealistic cat", clip, frames)

    assert pos[0][0].shape == (frames, SHORT_LEN, DIM)
    assert torch.allclose(pos[0][0], torch.full_like(pos[0][0], 7.0)), \
        "plain prompt was not held unchanged across every frame"
    assert neg[0][0].shape[0] == frames
    assert clip.encode_calls <= 2, \
        "a static prompt should be encoded once per positive/negative value"
    print("PASS plain prompt is reused across the whole batch")


def test_plain_multiline_prompt_is_not_treated_as_schedule():
    parsed = process_input_text("photorealistic cat\nsoft window light")
    assert parsed == {"0": "photorealistic cat\nsoft window light"}
    print("PASS plain multiline prompt remains plain text")


def test_malformed_declared_schedule_has_a_clear_error():
    try:
        process_input_text('"0": "cat", "8":')
    except ValueError as exc:
        assert "Invalid prompt schedule" in str(exc)
    else:
        raise AssertionError("malformed schedule was silently treated as prompt text")
    print("PASS malformed schedule reports a clear error")


def test_linear_mode_remains_unchanged_for_other_models():
    text = '"0" :"photorealistic cat", "4" :"painting"'

    pos, _ = schedule_conditioning(text, MockClip(), 5)
    tensor = pos[0][0]

    assert len(pos) == 1, "normal model scheduling unexpectedly gained timestep ranges"
    assert torch.allclose(tensor[0], torch.full_like(tensor[0], 7.0))
    assert torch.allclose(tensor[2], torch.full_like(tensor[2], 5.0))
    assert torch.allclose(tensor[4], torch.full_like(tensor[4], 3.0))
    print("PASS linear interpolation remains unchanged for other models")


def test_native_metadata_and_positive_negative_semantics_are_preserved():
    text = '"0" :"photorealistic cat --neg painting"'

    pos, neg = schedule_conditioning(text, MockClip(), 3)

    assert torch.allclose(pos[0][0], torch.full_like(pos[0][0], 7.0)), \
        "positive prompt embedding was altered or swapped"
    assert torch.allclose(neg[0][0], torch.full_like(neg[0][0], 3.0)), \
        "negative prompt embedding was altered or swapped"
    assert "attention_mask" in pos[0][1], "native attention_mask was discarded"
    assert pos[0][1]["attention_mask"].shape[0] == 3
    print("PASS native metadata and positive/negative semantics")


def test_hold_mode_uses_only_pure_keyframe_embeddings():
    text = '"0" :"photorealistic cat", "3" :"painting"'

    pos, _ = schedule_conditioning(
        text, MockClip(), 6, interpolation_mode="hold"
    )
    tensor = pos[0][0]

    assert torch.allclose(tensor[:3], torch.full_like(tensor[:3], 7.0)), \
        "frames before the keyframe must retain the pure first prompt"
    assert torch.allclose(tensor[3:], torch.full_like(tensor[3:], 3.0)), \
        "frames at and after the keyframe must use the pure second prompt"
    assert set(tensor.unique().tolist()) == {3.0, 7.0}, \
        "hold mode created an interpolated embedding"
    print("PASS hold mode uses only pure prompt embeddings")


def test_krea_transition_is_short_and_uses_pure_detail_conditioning():
    text = '"0" :"photorealistic cat", "8" :"painting"'

    pos, _ = schedule_conditioning(
        text, MockClip(), 12, interpolation_mode="krea_transition",
        timestep_split=0.775,
    )

    assert len(pos) == 2, "Krea transition should have structural and detail ranges"
    structural, detail = pos
    structural_tensor, structural_meta = structural
    detail_tensor, detail_meta = detail

    assert structural_meta["start_percent"] == 0.0
    assert structural_meta["end_percent"] == 0.775
    assert detail_meta["start_percent"] == 0.775
    assert detail_meta["end_percent"] == 1.0

    # The eight-frame interval gets a two-frame transition at frames 6 and 7.
    assert torch.allclose(
        structural_tensor[:6], torch.full_like(structural_tensor[:6], 7.0)
    ), "Krea blending started before the short transition window"
    assert 3.0 < structural_tensor[6].mean() < 7.0
    assert 3.0 < structural_tensor[7].mean() < 7.0
    assert torch.allclose(
        structural_tensor[8:], torch.full_like(structural_tensor[8:], 3.0)
    )

    # Final detail steps never receive an averaged Qwen embedding.
    assert torch.allclose(detail_tensor[6], torch.full_like(detail_tensor[6], 7.0))
    assert torch.allclose(detail_tensor[7], torch.full_like(detail_tensor[7], 3.0))
    assert set(detail_tensor.unique().tolist()) == {3.0, 7.0}
    print("PASS Krea transition uses short alpha ramp and pure detail prompts")


if __name__ == "__main__":
    test_varlen_schedule()
    test_uniform_schedule()
    test_plain_prompt_is_reused_for_the_whole_batch()
    test_plain_multiline_prompt_is_not_treated_as_schedule()
    test_malformed_declared_schedule_has_a_clear_error()
    test_linear_mode_remains_unchanged_for_other_models()
    test_native_metadata_and_positive_negative_semantics_are_preserved()
    test_hold_mode_uses_only_pure_keyframe_embeddings()
    test_krea_transition_is_short_and_uses_pure_detail_conditioning()
    print("All syntax_schedule tests passed.")
