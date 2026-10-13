import os
import json
import re
import base64
from typing import List, Dict, Optional

try:
    from .prm_validation import (
        PRM_MAX_ATTEMPTS,
        PRMValidationError,
        validate_grouped_step_coverage,
        validate_grouping,
        validation_feedback,
    )
except ImportError:
    from prm_validation import (
        PRM_MAX_ATTEMPTS,
        PRMValidationError,
        validate_grouped_step_coverage,
        validate_grouping,
        validation_feedback,
    )

ENABLE_FORCE_SPLIT = True
MAX_STEPS_LIMIT = 5

def clean_json_response(content: str) -> str:
    content = re.sub(r"```json\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```\s*", "", content)
    return content.strip()

def encode_image_to_base64(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

def get_next_image_filename(current_filename: str) -> Optional[str]:
    if not current_filename:
        return None
    match = re.search(r'(\d+)(?=\D*$)', current_filename)
    if match:
        num_str = match.group(1)
        next_val = int(num_str) + 1
        next_num_str = str(next_val).zfill(len(num_str))
        start, end = match.span(1)
        return current_filename[:start] + next_num_str + current_filename[end:]
    return None

def _chunk_steps(steps: List[Dict], limit: int) -> List[List[Dict]]:
    return [steps[i:i + limit] for i in range(0, len(steps), limit)]

async def async_group_steps(
    task_instruction: str,
    steps: List[Dict],
    async_call_llm_fn
) -> List[Dict]:
    if not steps:
        return []

    steps_text = ""
    for step in steps:
        steps_text += f"Step {step['step_id']}:\n{step['raw_content']}\n\n"

    system_prompt = """
    You are an expert in Hierarchical Reinforcement Learning.
    Decompose the GUI trajectory into logical subgoals based on the User Instruction.

    Return ONLY a valid JSON list. No extra text.

    Format:
    [
        {
            "subgoal_name": "Summary",
            "start_step": <int>,
            "end_step": <int>,
            "reasoning": "Why these steps are grouped"
        }
    ]
    """

    user_prompt = f"""
    ## User Task Instruction
    {task_instruction}

    ## Trajectory Steps
    {steps_text}

    Analyze the steps and group them. Return ONLY JSON.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    last_error = None
    for attempt in range(1, PRM_MAX_ATTEMPTS + 1):
        attempt_messages = messages + ([validation_feedback(last_error)] if last_error else [])
        json_result_str = await async_call_llm_fn(attempt_messages)

        try:
            if not json_result_str:
                raise PRMValidationError("grouping response must not be empty")
            subgoals_structure = validate_grouping(
                json.loads(clean_json_response(json_result_str)), len(steps)
            )
            enriched_subgoals = []
            for item in subgoals_structure:
                original_start = item['start_step']
                original_end = item['end_step']
                start_idx = original_start - 1
                end_idx = original_end

                current_segment = [
                    {
                        'step_idx': s['step_id'],
                        'content': s['raw_content'],
                        'image': s['image_file'],
                        'next_image': get_next_image_filename(s['image_file'])
                    }
                    for s in steps[start_idx:end_idx]
                ]

                context_step = None
                if start_idx > 0:
                    prev_s = steps[start_idx - 1]
                    context_step = {
                        'step_idx': prev_s['step_id'],
                        'content': f"[PREVIOUS CONTEXT] {prev_s['raw_content']}",
                        'image': prev_s['image_file'],
                        'next_image': get_next_image_filename(prev_s['image_file'])
                    }

                windows_data = []
                for window_idx, chunk in enumerate(_chunk_steps(current_segment, MAX_STEPS_LIMIT), start=1):
                    windows_data.append({
                        'window_idx': window_idx,
                        'start_step': chunk[0]['step_idx'],
                        'end_step': chunk[-1]['step_idx'],
                        'steps_data': chunk
                    })

                item['steps_data'] = ([context_step] if context_step else []) + current_segment
                item['windows_data'] = windows_data
                item['window_mode'] = 'multi' if len(windows_data) > 1 else 'single'
                enriched_subgoals.append(item)

            validate_grouped_step_coverage(
                enriched_subgoals, len(steps), require_windows=True
            )
            return enriched_subgoals
        except (json.JSONDecodeError, PRMValidationError, KeyError, TypeError, ValueError) as error:
            last_error = error
            print(
                f"[PRM Grouping] validation failed on attempt "
                f"{attempt}/{PRM_MAX_ATTEMPTS}: {error}"
            )

    return []
