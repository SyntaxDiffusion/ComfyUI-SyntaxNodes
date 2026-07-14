"""Self-contained FizzNodes-style batch prompt scheduling.

Ported from comfyui_fizznodes (BatchPromptSchedule path) with fixes for
variable-length text encoders (Qwen, Flux Krea, etc.) that produce different
token counts per prompt. No dependency on fizznodes or ComfyUI internals.
"""

from .ScheduleFuncs import ScheduleSettings, addWeighted, pad_with_zeros, process_input_text
from .BatchFuncs import (
    BatchPoolAnimConditioning,
    batch_split_weighted_subprompts,
    interpolate_prompt_seriesA,
    normalize_conditioning_tensor_shapes,
)


def _late_transition_weights(cur_prompts, next_prompts, max_window=8):
    """Return a short smoothstep blend before each prompt keyframe.

    At least the first frame of every interval remains on its pure prompt. The
    transition occupies roughly the final quarter of the interval, capped at
    ``max_window`` frames, rather than averaging embeddings for the whole span.
    """
    count = len(cur_prompts)
    weights = [1.0] * count
    start = 0
    while start < count:
        current = str(cur_prompts[start])
        upcoming = str(next_prompts[start])
        end = start + 1
        while (
            end < count
            and str(cur_prompts[end]) == current
            and str(next_prompts[end]) == upcoming
        ):
            end += 1

        interval = end - start
        if current != upcoming and interval > 1:
            target = max(2, (interval + 3) // 4)
            window = min(max_window, interval - 1, target)
            transition_start = end - window
            for offset, frame in enumerate(range(transition_start, end), start=1):
                alpha = offset / (window + 1)
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                weights[frame] = 1.0 - alpha
        start = end
    return weights


def _nearest_prompt_weights(blend_weights):
    """Choose the nearest pure endpoint for late detail-refinement steps."""
    return [1.0 if weight > 0.5 else 0.0 for weight in blend_weights]


def _set_timestep_range(conditioning, start, end):
    ranged = []
    for tensor, metadata in conditioning:
        ranged_metadata = metadata.copy()
        ranged_metadata["start_percent"] = float(start)
        ranged_metadata["end_percent"] = float(end)
        ranged.append([tensor, ranged_metadata])
    return ranged


def batch_prompt_schedule(settings: ScheduleSettings, clip, interpolation_mode="linear",
                          timestep_split=0.6):
    """Build batched (positive, negative) conditioning from a schedule string.

    Ported from fizznodes ScheduleTypes.batch_prompt_schedule.
    """
    animation_prompts = process_input_text(settings.text_g)

    pos, neg = batch_split_weighted_subprompts(animation_prompts, settings.pre_text_G, settings.app_text_G)

    pos_cur_prompt, pos_nxt_prompt, pos_weight = interpolate_prompt_seriesA(pos, settings)
    neg_cur_prompt, neg_nxt_prompt, neg_weight = interpolate_prompt_seriesA(neg, settings)

    if interpolation_mode == "hold":
        # Keep every conditioning on a real encoded prompt instead of creating
        # an embedding-space average. This is important for Krea2's stacked
        # Qwen hidden states, where interpolated tensors can drift off-manifold.
        pos_weight = [1.0] * settings.max_frames
        neg_weight = [1.0] * settings.max_frames
    elif interpolation_mode == "krea_transition":
        pos_weight = _late_transition_weights(pos_cur_prompt, pos_nxt_prompt)
        neg_weight = _late_transition_weights(neg_cur_prompt, neg_nxt_prompt)
    elif interpolation_mode != "linear":
        raise ValueError(f"Unknown prompt interpolation mode: {interpolation_mode}")

    encode_cache = {}
    p = BatchPoolAnimConditioning(
        pos_cur_prompt, pos_nxt_prompt, pos_weight, clip, settings, encode_cache
    )
    n = BatchPoolAnimConditioning(
        neg_cur_prompt, neg_nxt_prompt, neg_weight, clip, settings, encode_cache
    )

    if interpolation_mode == "krea_transition":
        # Blended Qwen states guide only the early structural portion. The final
        # steps use the nearest real prompt embedding to restore clean details.
        p_pure = BatchPoolAnimConditioning(
            pos_cur_prompt, pos_nxt_prompt, _nearest_prompt_weights(pos_weight),
            clip, settings, encode_cache,
        )
        n_pure = BatchPoolAnimConditioning(
            neg_cur_prompt, neg_nxt_prompt, _nearest_prompt_weights(neg_weight),
            clip, settings, encode_cache,
        )
        timestep_split = min(1.0, max(0.0, float(timestep_split)))
        p = _set_timestep_range(p, 0.0, timestep_split) + _set_timestep_range(p_pure, timestep_split, 1.0)
        n = _set_timestep_range(n, 0.0, timestep_split) + _set_timestep_range(n_pure, timestep_split, 1.0)

    p = normalize_conditioning_tensor_shapes(p)
    n = normalize_conditioning_tensor_shapes(n)

    return (p, n)


def schedule_conditioning(text, clip, max_frames, pre_text="", app_text="",
                          start_frame=0, end_frame=0,
                          pw_a=0.0, pw_b=0.0, pw_c=0.0, pw_d=0.0,
                          interpolation_mode="linear", timestep_split=0.6):
    """Convenience wrapper: schedule text -> batched (positive, negative) conditioning.

    `text` uses the FizzNodes format: "0" :"prompt", "10" :"other prompt"
    Negative prompts can be embedded with --neg inside each prompt.
    """
    settings = ScheduleSettings(
        text_g=text,
        pre_text_G=pre_text,
        app_text_G=app_text,
        text_L=None,
        pre_text_L=None,
        app_text_L=None,
        max_frames=max_frames,
        current_frame=None,
        print_output=False,
        pw_a=pw_a,
        pw_b=pw_b,
        pw_c=pw_c,
        pw_d=pw_d,
        start_frame=start_frame,
        end_frame=end_frame,
        width=None,
        height=None,
        crop_w=None,
        crop_h=None,
        target_width=None,
        target_height=None,
    )
    return batch_prompt_schedule(
        settings, clip, interpolation_mode=interpolation_mode,
        timestep_split=timestep_split,
    )
