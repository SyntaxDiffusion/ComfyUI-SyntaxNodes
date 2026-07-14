# These nodes were made using code from the Deforum extension for A1111 webui
# You can find the project here: https://github.com/deforum-art/sd-webui-deforum

import numexpr
import torch
import numpy as np
import pandas as pd
import re

from .ScheduleFuncs import *

def normalize_conditioning_tensor_shapes(conditioning_list):
    """
    Normalize tensor shapes in conditioning to ensure compatibility with control nets.
    Ensures all tensor dimensions are proper integers, not tuples or other types.
    Special handling for AnimateDiff evolved sampling and advanced control nets.
    """
    if not conditioning_list:
        return conditioning_list
    
    def fix_tensor_shape_access(tensor):
        """Fix tensor to ensure .shape[] returns integers instead of tuples"""
        if not hasattr(tensor, 'shape'):
            return tensor
            
        try:
            # Make tensor contiguous and clone to ensure proper shape representation
            fixed_tensor = tensor.contiguous()
            
            # Force tensor to have proper integer dimensions by creating a new view
            # This ensures that .shape[n] returns int instead of tuple
            shape = [int(dim) for dim in fixed_tensor.shape]
            fixed_tensor = fixed_tensor.view(*shape)
            
            # Ensure tensor is contiguous in memory
            if not fixed_tensor.is_contiguous():
                fixed_tensor = fixed_tensor.contiguous()
                
            return fixed_tensor
        except Exception as e:
            print(f"[normalize_conditioning_tensor_shapes] Warning: Failed to fix tensor shape: {e}")
            return tensor
    
    normalized_list = []
    for cond_item in conditioning_list:
        if isinstance(cond_item, list) and len(cond_item) >= 1:
            cond_tensor = cond_item[0]
            cond_dict = cond_item[1] if len(cond_item) > 1 else {}
            
            # Fix the main conditioning tensor
            cond_tensor = fix_tensor_shape_access(cond_tensor)
            
            # Handle pooled_output in the conditioning dict
            if 'pooled_output' in cond_dict and hasattr(cond_dict['pooled_output'], 'shape'):
                cond_dict['pooled_output'] = fix_tensor_shape_access(cond_dict['pooled_output'])
            
            # Handle any other tensors that might be in the conditioning dict
            for key, value in cond_dict.items():
                if hasattr(value, 'shape') and hasattr(value, 'contiguous'):
                    try:
                        cond_dict[key] = fix_tensor_shape_access(value)
                    except:
                        # If fixing fails, leave the original value
                        pass
            
            normalized_list.append([cond_tensor, cond_dict])
        else:
            normalized_list.append(cond_item)
    
    return normalized_list

# calculates numexpr expressions from the text input and return a string
def prepare_batch_prompt(prompt_series, max_frames, frame_idx, prompt_weight_1=0, prompt_weight_2=0, prompt_weight_3=0,
                         prompt_weight_4=0):
    max_f = max_frames - 1
    pattern = r'`.*?`'  # set so the expression will be read between two backticks (``)
    regex = re.compile(pattern)
    prompt_parsed = str(prompt_series)

    for match in regex.finditer(prompt_parsed):
        matched_string = match.group(0)
        parsed_string = matched_string.replace('t', f'{frame_idx}').replace("pw_a", f"{prompt_weight_1}").replace("pw_b",
                                                    f"{prompt_weight_2}").replace("pw_c", f"{prompt_weight_3}").replace("pw_d",
                                                    f"{prompt_weight_4}").replace("max_f",
                                                    f"{max_f}").replace('`', '')  # replace t, max_f and `` respectively
        parsed_value = numexpr.evaluate(parsed_string)
        prompt_parsed = prompt_parsed.replace(matched_string, str(parsed_value))
    return prompt_parsed.strip()

def prepare_batch_promptA(prompt, settings:ScheduleSettings, index):
    max_f = settings.max_frames - 1
    pattern = r'`.*?`'  # set so the expression will be read between two backticks (``)
    regex = re.compile(pattern)
    prompt_parsed = str(prompt)

    for match in regex.finditer(prompt_parsed):
        matched_string = match.group(0)
        parsed_string = matched_string.replace(
            't',
            f'{index}').replace("pw_a",
            f"{settings.pw_a[index]}").replace("pw_b",
            f"{settings.pw_b[index]}").replace("pw_c",
            f"{settings.pw_c[index]}").replace("pw_d",
            f"{settings.pw_d[index]}").replace("max_f",
            f"{max_f}").replace('`', '')  # replace t, max_f and `` respectively
        parsed_value = numexpr.evaluate(parsed_string)
        prompt_parsed = prompt_parsed.replace(matched_string, str(parsed_value))
    return prompt_parsed.strip()

#splits the prompt into positive and negative outputs
#denoted with --neg for where the split should be.
def batch_split_weighted_subprompts(text, pre_text, app_text):
    pos = {}
    neg = {}
    pre_text = str(pre_text)
    app_text = str(app_text)

    if "--neg" in pre_text:
        pre_pos, pre_neg = pre_text.split("--neg")
    else:
        pre_pos, pre_neg = pre_text, ""

    if "--neg" in app_text:
        app_pos, app_neg = app_text.split("--neg")
    else:
        app_pos, app_neg = app_text, ""

    for frame, prompt in text.items():
        negative_prompts = ""
        positive_prompts = ""
        prompt_split = prompt.split("--neg")

        if len(prompt_split) > 1:
            positive_prompts, negative_prompts = prompt_split[0], prompt_split[1]
        else:
            positive_prompts = prompt_split[0]

        pos[frame] = ""
        neg[frame] = ""
        pos[frame] += (str(pre_pos) + " " + positive_prompts + " " + str(app_pos))
        neg[frame] += (str(pre_neg) + " " + negative_prompts + " " + str(app_neg))
        if pos[frame].endswith('0'):
            pos[frame] = pos[frame][:-1]
        if neg[frame].endswith('0'):
            neg[frame] = neg[frame][:-1]
    return pos, neg

# converts the prompt weight variables to tuples. if it is an int variable,
# set all frames to have the same value
def convert_pw_to_tuples(settings):
    if isinstance(settings.pw_a, (int, float, np.float64)):
        settings.pw_a = tuple([settings.pw_a] * settings.max_frames)
    if isinstance(settings.pw_b, (int, float, np.float64)):
        settings.pw_b = tuple([settings.pw_b] * settings.max_frames)
    if isinstance(settings.pw_c, (int, float, np.float64)):
        settings.pw_c = tuple([settings.pw_c] * settings.max_frames)
    if isinstance(settings.pw_d, (int, float, np.float64)):
        settings.pw_d = tuple([settings.pw_d] * settings.max_frames)


def interpolate_prompt_seriesA(animation_prompts, settings:ScheduleSettings):

    max_f = settings.max_frames  # needed for numexpr even though it doesn't look like it's in use.
    parsed_animation_prompts = {}


    for key, value in animation_prompts.items():
        if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
            parsed_animation_prompts[key] = value
        else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
            parsed_animation_prompts[int(numexpr.evaluate(key))] = value

    sorted_prompts = sorted(parsed_animation_prompts.items(), key=lambda item: int(item[0]))

    # Automatically set the first keyframe to 0 if it's missing
    if sorted_prompts[0][0] != "0":
        sorted_prompts.insert(0, ("0", sorted_prompts[0][1]))

    # Automatically set the last keyframe to the maximum number of frames
    if sorted_prompts[-1][0] != str(settings.max_frames):
        sorted_prompts.append((str(settings.max_frames), sorted_prompts[-1][1]))

    # Setup containers for interpolated prompts
    nan_list = [np.nan for a in range(settings.max_frames)]
    cur_prompt_series = pd.Series(nan_list,dtype=object)
    nxt_prompt_series = pd.Series(nan_list,dtype=object)

    # simple array for strength values
    weight_series = [np.nan] * settings.max_frames

    # in case there is only one keyed prompt, set all prompts to that prompt
    if settings.max_frames == 1:
        for i in range(0, len(cur_prompt_series) - 1):
            current_prompt = sorted_prompts[0][1]
            cur_prompt_series[i] = str(current_prompt)
            nxt_prompt_series[i] = str(current_prompt)

    #make sure prompt weights are tuples and convert them if not
    convert_pw_to_tuples(settings)

    # Initialized outside of loop for nan check
    current_key = 0
    next_key = 0

    # For every keyframe prompt except the last
    for i in range(0, len(sorted_prompts) - 1):
        # Get current and next keyframe
        current_key = int(sorted_prompts[i][0])
        next_key = int(sorted_prompts[i + 1][0])

        # Ensure there's no weird ordering issues or duplication in the animation prompts
        # (unlikely because we sort above, and the json parser will strip dupes)
        if current_key >= next_key:
            print(
                f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_key} are not monotonously increasing; skipping interpolation.")
            continue

        # Get current and next keyframes' positive and negative prompts (if any)
        current_prompt = sorted_prompts[i][1]
        next_prompt = sorted_prompts[i + 1][1]

        # Calculate how much to shift the weight from current to next prompt at each frame.
        weight_step = 1 / (next_key - current_key)

        for f in range(max(current_key, 0), min(next_key, len(cur_prompt_series))):
            next_weight = weight_step * (f - current_key)
            current_weight = 1 - next_weight

            # add the appropriate prompts and weights to their respective containers.
            weight_series[f] = 0.0
            cur_prompt_series[f] = str(current_prompt)
            nxt_prompt_series[f] = str(next_prompt)

            weight_series[f] += current_weight

        current_key = next_key
        next_key = settings.max_frames
        current_weight = 0.0

    index_offset = 0

    # Evaluate the current and next prompt's expressions
    for i in range(settings.start_frame, min(settings.end_frame,len(cur_prompt_series))):
        cur_prompt_series[i] = prepare_batch_promptA(cur_prompt_series[i], settings, i)
        nxt_prompt_series[i] = prepare_batch_promptA(nxt_prompt_series[i], settings, i)
        if settings.print_output == True:
            # Show the to/from prompts with evaluated expressions for transparency.
            if(settings.start_frame >= i):
                if(settings.end_frame > 0):
                    if(settings.end_frame > i):
                        print("\n", "Max Frames: ", settings.max_frames, "\n", "frame index: ", (settings.start_frame + i),
                              "\n", "Current Prompt: ",
                              cur_prompt_series[i], "\n", "Next Prompt: ", nxt_prompt_series[i], "\n", "Strength : ",
                              weight_series[i], "\n")
                else:
                    print("\n", "Max Frames: ", settings.max_frames, "\n", "frame index: ", (settings.start_frame + i), "\n", "Current Prompt: ",
                          cur_prompt_series[i], "\n", "Next Prompt: ", nxt_prompt_series[i], "\n", "Strength : ",
                          weight_series[i], "\n")
        index_offset = index_offset + 1

    # Output methods depending if the prompts are the same or if the current frame is a keyframe.
    # if it is an in-between frame and the prompts differ, composable diffusion will be performed.
    return (cur_prompt_series, nxt_prompt_series, weight_series)

def interpolate_prompt_series(animation_prompts, max_frames, start_frame, pre_text, app_text, prompt_weight_1=[],
                              prompt_weight_2=[], prompt_weight_3=[], prompt_weight_4=[], Is_print = False):

    max_f = max_frames  # needed for numexpr even though it doesn't look like it's in use.
    parsed_animation_prompts = {}


    for key, value in animation_prompts.items():
        if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
            parsed_animation_prompts[key] = value
        else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
            parsed_animation_prompts[int(numexpr.evaluate(key))] = value

    sorted_prompts = sorted(parsed_animation_prompts.items(), key=lambda item: int(item[0]))

    # Automatically set the first keyframe to 0 if it's missing
    if sorted_prompts[0][0] != "0":
        sorted_prompts.insert(0, ("0", sorted_prompts[0][1]))

    # Automatically set the last keyframe to the maximum number of frames
    if sorted_prompts[-1][0] != str(max_frames):
        sorted_prompts.append((str(max_frames), sorted_prompts[-1][1]))
    # Setup containers for interpolated prompts
    cur_prompt_series = pd.Series([np.nan for a in range(max_frames)])
    nxt_prompt_series = pd.Series([np.nan for a in range(max_frames)])

    # simple array for strength values
    weight_series = [np.nan] * max_frames

    # in case there is only one keyed promt, set all prompts to that prompt
    if len(sorted_prompts) == 1:
        for i in range(0, len(cur_prompt_series) - 1):
            current_prompt = sorted_prompts[0][1]
            cur_prompt_series[i] = str(current_prompt)
            nxt_prompt_series[i] = str(current_prompt)

    # Initialized outside of loop for nan check
    current_key = 0
    next_key = 0

    if type(prompt_weight_1) in {int, float, np.float64}:
       prompt_weight_1 = tuple([prompt_weight_1] * max_frames)

    if type(prompt_weight_2) in {int, float, np.float64}:
       prompt_weight_2 = tuple([prompt_weight_2] * max_frames)

    if type(prompt_weight_3) in {int, float, np.float64}:
       prompt_weight_3 = tuple([prompt_weight_3] * max_frames)

    if type(prompt_weight_4) in {int, float, np.float64}:
       prompt_weight_4 = tuple([prompt_weight_4] * max_frames)

    # For every keyframe prompt except the last
    for i in range(0, len(sorted_prompts) - 1):
        # Get current and next keyframe
        current_key = int(sorted_prompts[i][0])
        next_key = int(sorted_prompts[i + 1][0])

        # Ensure there's no weird ordering issues or duplication in the animation prompts
        # (unlikely because we sort above, and the json parser will strip dupes)
        if current_key >= next_key:
            print(
                f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_key} are not monotonously increasing; skipping interpolation.")
            continue

        # Get current and next keyframes' positive and negative prompts (if any)
        current_prompt = sorted_prompts[i][1]
        next_prompt = sorted_prompts[i + 1][1]

        # Calculate how much to shift the weight from current to next prompt at each frame.
        weight_step = 1 / (next_key - current_key)

        for f in range(max(current_key, 0), min(next_key, len(cur_prompt_series))):
            next_weight = weight_step * (f - current_key)
            current_weight = 1 - next_weight

            # add the appropriate prompts and weights to their respective containers.
            weight_series[f] = 0.0
            cur_prompt_series[f] = str(current_prompt)
            nxt_prompt_series[f] = str(next_prompt)

            weight_series[f] += current_weight

        current_key = next_key
        next_key = max_frames
        current_weight = 0.0

    index_offset = 0



    # Evaluate the current and next prompt's expressions
    for i in range(start_frame, len(cur_prompt_series)):
        cur_prompt_series[i] = prepare_batch_prompt(cur_prompt_series[i], max_frames, i, prompt_weight_1[i],
                                                    prompt_weight_2[i], prompt_weight_3[i], prompt_weight_4[i])
        nxt_prompt_series[i] = prepare_batch_prompt(nxt_prompt_series[i], max_frames, i, prompt_weight_1[i],
                                                    prompt_weight_2[i], prompt_weight_3[i], prompt_weight_4[i])

        if Is_print == True:
            # Show the to/from prompts with evaluated expressions for transparency.
            print("\n", "Max Frames: ", max_frames, "\n", "frame index: ", (start_frame + i), "\n", "Current Prompt: ",
                  cur_prompt_series[i], "\n", "Next Prompt: ", nxt_prompt_series[i], "\n", "Strength : ",
                  weight_series[i], "\n")
        index_offset = index_offset + 1



    # Output methods depend on if the prompts are the same or if the current frame is a keyframe.
    # if it is an in-between frame and the prompts differ, composable diffusion will be performed.
    return (cur_prompt_series, nxt_prompt_series, weight_series)

def _encode_cached(clip, prompt, cache):
    """Encode each unique prompt string once. The text encoder forward is the
    dominant cost here - Krea2/Qwen run a multi-billion-parameter LLM per
    encode, and schedules repeat the same few prompts across hundreds of frames.
    """
    key = str(prompt)
    if key not in cache:
        tokens = clip.tokenize(key)
        try:
            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        except TypeError:
            # Compatibility with older ComfyUI versions and lightweight test
            # doubles that predate return_dict.
            output = clip.encode_from_tokens(tokens, return_pooled=True)

        if isinstance(output, dict):
            metadata = output.copy()
            cond = metadata.pop("cond")
        elif isinstance(output, tuple):
            cond, pooled = output
            metadata = {"pooled_output": pooled} if pooled is not None else {}
        else:
            cond = output
            metadata = {}
        cache[key] = (cond, metadata)
    return cache[key]


def _batch_conditioning_metadata(metadata_frames):
    """Batch native CLIP metadata without discarding model-specific fields."""
    if not metadata_frames:
        return {}

    result = {}
    keys = set().union(*(metadata.keys() for metadata in metadata_frames))
    for key in keys:
        values = [metadata.get(key) for metadata in metadata_frames]
        present = [value for value in values if value is not None]
        if not present:
            continue

        if all(torch.is_tensor(value) for value in present) and len(present) == len(values):
            normalized = present
            if "mask" in key.lower() and all(value.ndim == 2 for value in present):
                max_length = max(value.shape[1] for value in present)
                normalized = [
                    torch.nn.functional.pad(value, (0, max_length - value.shape[1]))
                    if value.shape[1] < max_length else value
                    for value in present
                ]
            elif len({tuple(value.shape[1:]) for value in present}) != 1:
                # Extra embedding tensors generally use B,S,D. Repeat-pad the
                # sequence axis just like the primary conditioning tensor.
                if all(value.ndim == 3 for value in present) and len({value.shape[2] for value in present}) == 1:
                    max_length = max(value.shape[1] for value in present)
                    normalized = [pad_with_zeros(value, max_length)[0] for value in present]
                else:
                    continue

            if all(value.shape[0] == 1 for value in normalized):
                result[key] = torch.cat(normalized, dim=0)
            elif all(value.shape == normalized[0].shape for value in normalized):
                result[key] = normalized[0]
            continue

        if any(torch.is_tensor(value) for value in present):
            continue

        first = present[0]
        if all(value is first or value == first for value in present[1:]):
            result[key] = first

    return result


def BatchPoolAnimConditioning(cur_prompt_series, nxt_prompt_series, weight_series, clip, settings:ScheduleSettings,
                              encode_cache=None):
    cond_out = []
    metadata_out = []
    max_size = 0
    cond_dim = None  # Track the conditioning dimension
    print(f"[BatchPoolAnimConditioning] Starting with {len(cur_prompt_series)} prompts")

    # Calculate max size and dimension first
    # Use actual end_frame or max_frames, whichever is valid
    actual_end_frame = settings.end_frame if settings.end_frame > 0 else settings.max_frames
    actual_start_frame = settings.start_frame

    encode_cache = {} if encode_cache is None else encode_cache
    if max_size == 0:
        for i in range(actual_start_frame, min(actual_end_frame, len(cur_prompt_series))):
            # Scan both series: a nxt prompt can be longer than every cur prompt
            for series in (cur_prompt_series, nxt_prompt_series):
                if i < len(series) and series[i] is not None:
                    cond_s, _ = _encode_cached(clip, series[i], encode_cache)
                    max_size = max(max_size, cond_s.shape[1])
                    if cond_dim is None:
                        cond_dim = cond_s.shape[-1]  # Get the last dimension (768 for SD, 3584 for Qwen, etc)

    for i in range(actual_start_frame, min(actual_end_frame, len(cur_prompt_series))):
        # Ensure we have valid prompts at this index
        if i < len(cur_prompt_series) and cur_prompt_series[i] is not None:
            cond_to, metadata_to = _encode_cached(clip, cur_prompt_series[i], encode_cache)
        else:
            # If no current prompt, skip this frame
            continue

        if i < len(nxt_prompt_series) and nxt_prompt_series[i] is not None:
            cond_from, metadata_from = _encode_cached(clip, nxt_prompt_series[i], encode_cache)
        else:
            # No next prompt: blend the current prompt with itself (a no-op)
            # rather than with zeros, which dims the conditioning signal
            cond_from = cond_to
            metadata_from = metadata_to

        # Get the weight for this frame
        current_weight = weight_series[i] if i < len(weight_series) else 1.0

        if current_weight >= 1.0:
            # Preserve the native prompt tensor and all of its metadata exactly.
            # Running a nominal 100/0 blend still pads and performs arithmetic,
            # which is unnecessary and risky for model-specific conditioning.
            interpolated_cond = cond_to
            interpolated_metadata = metadata_to.copy()
        elif current_weight <= 0.0:
            interpolated_cond = cond_from
            interpolated_metadata = metadata_from.copy()
        else:
            interpolated_conditioning = addWeighted([[cond_to, metadata_to]],
                                                    [[cond_from, metadata_from]],
                                                    current_weight, max_size)
            interpolated_cond = interpolated_conditioning[0][0]
            interpolated_metadata = interpolated_conditioning[0][1]

        cond_out.append(interpolated_cond)
        metadata_out.append(interpolated_metadata)

    # Only concatenate if we have valid outputs
    if cond_out:
        # Variable-length encoders (Qwen, Krea, etc.) can produce longer conds
        # from nxt prompts than the max_size pre-scan of cur prompts detected,
        # so pad everything to the batch max before concatenating.
        batch_max = max(c.shape[1] for c in cond_out)
        cond_out = [pad_with_zeros(c, batch_max)[0] for c in cond_out]
        print(f"[BatchPoolAnimConditioning] Concatenating {len(cond_out)} conditioning tensors (seq len {batch_max})")
        final_conditioning = torch.cat(cond_out, dim=0)
        print(f"[BatchPoolAnimConditioning] Final conditioning shape: {final_conditioning.shape}")
        final_metadata = _batch_conditioning_metadata(metadata_out)
        if "pooled_output" in final_metadata:
            print(f"[BatchPoolAnimConditioning] Final pooled output shape: {final_metadata['pooled_output'].shape}")
        if "attention_mask" in final_metadata:
            print(f"[BatchPoolAnimConditioning] Final attention mask shape: {final_metadata['attention_mask'].shape}")
        return [[final_conditioning, final_metadata]]
    else:
        # Return empty conditioning if no valid frames - use detected dimension or default
        if cond_dim is None:
            cond_dim = 3584  # Default to Qwen dimension if we couldn't detect
        return [[torch.zeros((1, 1, cond_dim)), {}]]
def BatchGLIGENConditioning(cur_prompt_series, nxt_prompt_series, weight_series, clip):
    pooled_out = []
    cond_out = []
    max_size = 0
    if max_size == 0:
        for i in range(len(cur_prompt_series)):
            tokens = clip.tokenize(str(cur_prompt_series[i]))
            result = clip.encode_from_tokens(tokens, return_pooled=True)
            # Handle both tuple return and single tensor return
            if isinstance(result, tuple):
                cond_to, pooled_to = result
            else:
                cond_to = result
                pooled_to = None
            tensor_size = cond_to.shape[1]
            max_size = max(max_size, tensor_size)

    for i in range(len(cur_prompt_series)):
        tokens = clip.tokenize(str(cur_prompt_series[i]))
        result = clip.encode_from_tokens(tokens, return_pooled=True)
        # Handle both tuple return and single tensor return
        if isinstance(result, tuple):
            cond_to, pooled_to = result
        else:
            cond_to = result
            pooled_to = None

        tokens = clip.tokenize(str(nxt_prompt_series[i]))
        result = clip.encode_from_tokens(tokens, return_pooled=True)
        # Handle both tuple return and single tensor return
        if isinstance(result, tuple):
            cond_from, pooled_from = result
        else:
            cond_from = result
            pooled_from = None

        # Build conditioning dicts with pooled_output only if available
        cond_to_dict = {"pooled_output": pooled_to} if pooled_to is not None else {}
        cond_from_dict = {"pooled_output": pooled_from} if pooled_from is not None else {}

        interpolated_conditioning = addWeighted([[cond_to, cond_to_dict]],
                                                [[cond_from, cond_from_dict]],
                                                weight_series[i], max_size)

        interpolated_cond = interpolated_conditioning[0][0]
        if pooled_to is not None or pooled_from is not None:
            interpolated_pooled = interpolated_conditioning[0][1].get("pooled_output", pooled_from if pooled_from is not None else pooled_to)
            pooled_out.append(interpolated_pooled)

        cond_out.append(interpolated_cond)

    # Only concatenate pooled outputs if we have any
    if pooled_out:
        final_pooled_output = torch.cat(pooled_out, dim=0)
    else:
        final_pooled_output = None

    # Pad to the batch max seq length for variable-length encoders
    if cond_out:
        batch_max = max(c.shape[1] for c in cond_out)
        cond_out = [pad_with_zeros(c, batch_max)[0] for c in cond_out]
    final_conditioning = torch.cat(cond_out, dim=0) if cond_out else None

    return cond_out, pooled_out

def BatchPoolAnimConditioningSDXL(cur_prompt_series, nxt_prompt_series, weight_series, clip, settings:ScheduleSettings):
    pooled_out = []
    cond_out = []
    max_size = 0
    cond_dim = None  # Track the conditioning dimension
    
    # Use actual end_frame or max_frames, whichever is valid
    actual_end_frame = settings.end_frame if settings.end_frame > 0 else settings.max_frames
    actual_start_frame = settings.start_frame
    
    if max_size == 0:
        for i in range(actual_start_frame, min(actual_end_frame, len(cur_prompt_series))):
            if i < len(cur_prompt_series) and cur_prompt_series[i] is not None:
                tokens = clip.tokenize(str(cur_prompt_series[i]))
                result = clip.encode_from_tokens(tokens, return_pooled=True)
                # Handle both tuple return and single tensor return
                if isinstance(result, tuple):
                    cond_to, pooled_to = result
                else:
                    cond_to = result
                max_size = max(max_size, cond_to.shape[1])
                if cond_dim is None:
                    cond_dim = cond_to.shape[-1]  # Get the last dimension
                
    for i in range(actual_start_frame, min(actual_end_frame, len(cur_prompt_series))):
        interpolated_conditioning = addWeighted(cur_prompt_series[i],
                                                nxt_prompt_series[i],
                                                weight_series[i])

        interpolated_cond = interpolated_conditioning[0][0]
        interpolated_pooled = interpolated_conditioning[0][1].get("pooled_output") if len(interpolated_conditioning[0]) > 1 else None

        if interpolated_pooled is not None:
            pooled_out.append(interpolated_pooled)
        cond_out.append(interpolated_cond)

    # Only concatenate if we have valid outputs
    if cond_out:
        # Pad to the batch max seq length for variable-length encoders
        batch_max = max(c.shape[1] for c in cond_out)
        cond_out = [pad_with_zeros(c, batch_max)[0] for c in cond_out]
        print(f"[BatchPoolAnimConditioningSDXL] Concatenating {len(cond_out)} conditioning tensors (seq len {batch_max})")
        final_conditioning = torch.cat(cond_out, dim=0)
        print(f"[BatchPoolAnimConditioningSDXL] Final conditioning shape: {final_conditioning.shape}")
        # Only include pooled_output if we have any
        if pooled_out:
            final_pooled_output = torch.cat(pooled_out, dim=0)
            print(f"[BatchPoolAnimConditioningSDXL] Final pooled output shape: {final_pooled_output.shape}")
            return [[final_conditioning, {"pooled_output": final_pooled_output}]]
        else:
            print(f"[BatchPoolAnimConditioningSDXL] No pooled output")
            return [[final_conditioning, {}]]
    else:
        # Return empty conditioning if no valid frames - use detected dimension or default
        if cond_dim is None:
            cond_dim = 3584  # Default to Qwen dimension if we couldn't detect
        return [[torch.zeros((1, 1, cond_dim)), {}]]


def BatchInterpolatePromptsSDXL(animation_promptsG, animation_promptsL, clip, settings: ScheduleSettings):
    convert_pw_to_tuples(settings)
    # parse the conditioning strength and determine in-betweens.
    # Get prompts sorted by keyframe
    max_f = settings.max_frames  # needed for numexpr even though it doesn't look like it's in use.
    parsed_animation_promptsG = {}
    parsed_animation_promptsL = {}

    for key, value in animation_promptsG.items():
        if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
            parsed_animation_promptsG[key] = value
        else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
            parsed_animation_promptsG[int(numexpr.evaluate(key))] = value

    sorted_prompts_G = sorted(parsed_animation_promptsG.items(), key=lambda item: int(item[0]))

    for key, value in animation_promptsL.items():
        if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
            parsed_animation_promptsL[key] = value
        else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
            parsed_animation_promptsL[int(numexpr.evaluate(key))] = value

    sorted_prompts_L = sorted(parsed_animation_promptsL.items(), key=lambda item: int(item[0]))

    # Setup containers for interpolated prompts
    cur_prompt_series_G = pd.Series([np.nan for a in range(settings.max_frames)])
    nxt_prompt_series_G = pd.Series([np.nan for a in range(settings.max_frames)])

    cur_prompt_series_L = pd.Series([np.nan for a in range(settings.max_frames)])
    nxt_prompt_series_L = pd.Series([np.nan for a in range(settings.max_frames)])

    # simple array for strength values
    weight_series = [np.nan] * settings.max_frames

    def constructPrompt(sorted_prompts, cur_prompt, nxt_prompt, pre_text, app_text):
        if len(sorted_prompts) - 1 == 0:
            for i in range(0, len(sorted_prompts) - 1):
                current_prompt = sorted_prompts[0][1]
                cur_prompt[i] = str(pre_text) + " " + str(current_prompt) + " " + str(app_text)
                nxt_prompt[i] = str(pre_text) + " " + str(current_prompt) + " " + str(app_text)
        return cur_prompt, nxt_prompt

    # in case there is only one keyed promt, set all prompts to that prompt
    cur_prompt_series_G, nxt_prompt_series_G = constructPrompt(sorted_prompts_G, cur_prompt_series_G,
                                                               nxt_prompt_series_G, settings.pre_text_G,
                                                               settings.app_text_G)
    cur_prompt_series_L, nxt_prompt_series_L = constructPrompt(sorted_prompts_L, cur_prompt_series_L,
                                                               nxt_prompt_series_L, settings.pre_text_L,
                                                               settings.app_text_L)
    if len(sorted_prompts_L) - 1 == 0:
        for i in range(0, len(cur_prompt_series_L) - 1):
            current_prompt_L = sorted_prompts_L[0][1]
            cur_prompt_series_L[i] = str(pre_text_L) + " " + str(current_prompt_L) + " " + str(app_text_L)
            nxt_prompt_series_L[i] = str(pre_text_L) + " " + str(current_prompt_L) + " " + str(app_text_L)

    # Initialized outside of loop for nan check
    current_key = 0
    next_key = 0

    # For every keyframe prompt except the last
    for i in range(0, len(sorted_prompts_G) - 1):
        # Get current and next keyframe
        current_key = int(sorted_prompts_G[i][0])
        next_key = int(sorted_prompts_G[i + 1][0])

        # Ensure there's no weird ordering issues or duplication in the animation prompts
        # (unlikely because we sort above, and the json parser will strip dupes)
        if current_key >= next_key:
            print(
                f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_key} are not monotonously increasing; skipping interpolation.")
            continue

        # Get current and next keyframes' positive and negative prompts (if any)
        current_prompt_G = sorted_prompts_G[i][1]
        next_prompt_G = sorted_prompts_G[i + 1][1]

        # Calculate how much to shift the weight from current to next prompt at each frame.
        weight_step = 1 / (next_key - current_key)

        for f in range(current_key, next_key):
            next_weight = weight_step * (f - current_key)
            current_weight = 1 - next_weight

            # add the appropriate prompts and weights to their respective containers.
            if f < settings.max_frames:
                cur_prompt_series_G[f] = ''
                nxt_prompt_series_G[f] = ''
                weight_series[f] = 0.0

                cur_prompt_series_G[f] += (str(current_prompt_G))
                nxt_prompt_series_G[f] += (str(next_prompt_G))

                weight_series[f] += current_weight

        current_key = next_key
        next_key = settings.max_frames
        current_weight = 0.0
        # second loop to catch any nan runoff
        for f in range(current_key, next_key):
            next_weight = weight_step * (f - current_key)

            # add the appropriate prompts and weights to their respective containers.
            cur_prompt_series_G[f] = ''
            nxt_prompt_series_G[f] = ''
            weight_series[f] = current_weight

            cur_prompt_series_G[f] = (str(current_prompt_G))
            nxt_prompt_series_G[f] = (str(next_prompt_G))

    # Reset outside of loop for nan check
    current_key = 0
    next_key = 0

    for i in range(0, len(sorted_prompts_L) - 1):

        current_key = int(sorted_prompts_L[i][0])
        next_key = int(sorted_prompts_L[i + 1][0])

        # Ensure there's no weird ordering issues or duplication in the animation prompts
        # (unlikely because we sort above, and the json parser will strip dupes)
        if current_key >= next_key:
            print(
                f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_key} are not monotonously increasing; skipping interpolation.")
            continue

        # Get current and next keyframes' positive and negative prompts (if any)
        current_prompt_L = sorted_prompts_L[i][1]
        next_prompt_L = sorted_prompts_L[i + 1][1]

        # Calculate how much to shift the weight from current to next prompt at each frame.
        weight_step = 1 / (next_key - current_key)

        for f in range(current_key, next_key):
            next_weight = weight_step * (f - current_key)
            current_weight = 1 - next_weight

            # add the appropriate prompts and weights to their respective containers.
            if f < settings.max_frames:
                cur_prompt_series_L[f] = ''
                nxt_prompt_series_L[f] = ''
                weight_series[f] = 0.0

                cur_prompt_series_L[f] += (str(current_prompt_L))
                nxt_prompt_series_L[f] += (str(next_prompt_L))

                weight_series[f] += current_weight

        current_key = next_key
        next_key = settings.max_frames
        current_weight = 0.0
        # second loop to catch any nan runoff
        for f in range(current_key, next_key):
            next_weight = weight_step * (f - current_key)
            # add the appropriate prompts and weights to their respective containers.
            cur_prompt_series_L[f] = ''
            nxt_prompt_series_L[f] = ''
            weight_series[f] = current_weight

            cur_prompt_series_L[f] += (str(current_prompt_L))
            nxt_prompt_series_L[f] += (str(next_prompt_L))

    # Evaluate the current and next prompt's expressions
    for i in range(0, settings.max_frames):
        cur_prompt_series_G[i] = prepare_batch_promptA(cur_prompt_series_G[i], settings, i)
        nxt_prompt_series_G[i] = prepare_batch_promptA(nxt_prompt_series_G[i], settings, i)
        cur_prompt_series_L[i] = prepare_batch_promptA(cur_prompt_series_L[i], settings, i)
        nxt_prompt_series_L[i] = prepare_batch_promptA(nxt_prompt_series_L[i], settings, i)

    current_conds = []
    next_conds = []

    for i in range(0, settings.max_frames):
        current_conds.append(SDXLencode(cur_prompt_series_G[i], cur_prompt_series_L[i], settings, clip))
        next_conds.append(SDXLencode(nxt_prompt_series_L[i], nxt_prompt_series_L[i], settings, clip))


    if settings.print_output == True:
        # Show the to/from prompts with evaluated expressions for transparency.
        for i in range(0, settings.max_frames):
            print("\n", "Max Frames: ", settings.max_frames, "Curr Frame: ", i, "\n", "Current Prompt G: ",
                  cur_prompt_series_G[i],
                  "\n", "Current Prompt L: ", cur_prompt_series_L[i], "\n", "Next Prompt G: ", nxt_prompt_series_G[i],
                  "\n", "Next Prompt L : ", nxt_prompt_series_L[i], "\n", "Current weight: ", weight_series[i])

    return current_conds, next_conds, weight_series
