import os
import json
import re
import base64
from typing import List, Dict, Optional, Tuple

try:
    from .prm_validation import (
        PRM_MAX_ATTEMPTS,
        PRMValidationError,
        collect_original_group_conclusions,
        validate_group_conclusion,
        validate_phase3_refinement,
        validate_step_critics,
        validate_task_reference,
        validation_feedback,
    )
except ImportError:
    from prm_validation import (
        PRM_MAX_ATTEMPTS,
        PRMValidationError,
        collect_original_group_conclusions,
        validate_group_conclusion,
        validate_phase3_refinement,
        validate_step_critics,
        validate_task_reference,
        validation_feedback,
    )

MAX_STEPS_PER_WINDOW = 5

CORE_SCORING_PROMPT = """
Core scoring principle:
- Thought/agent response defines the intended action; it does not prove execution.
- Screenshots before/after prove execution through visual state changes.
- Task instructions define whether the resulting content/value/layout is correct.
- Later available context can help judge whether a state change was useful, reused, or a dead end; do not assume unseen visual details from text-only context.
- Previous Group Results provide concise historical context, but they do not prove the current group is correct.

Evidence priority:
1. Use screenshots when provided to judge what actually happened; otherwise use the verified step-level evidence summaries.
2. Use thought/response/action text to infer what the agent intended.
3. Use the next screenshot/next response and later steps to judge whether the effect mattered.
4. If text and visuals conflict, trust visuals for execution and mention the conflict.
5. If evidence is insufficient, mark execution_result as "Uncertain" instead of hallucinating.

Forward scoring usage:
- Score from task rules, screenshots, actions, and previous group results.
- Step scores judge local execution and subgoal contribution; group score judges subgoal status and goal contribution.
- A visible UI change is not enough for high reward. The changed value, label, line break, separator, cell, field, range, order, and format must satisfy the task/subgoal constraints.
"""

PHASE0_CONSTRAINTS = """
Phase 0: Task Reference
- Purpose: build the reference used by all later judgments.
- Build a rule-consistent reference before judging steps or groups.
- Step text may reveal candidate objects, values, formulas, routes, or assumptions, but it does not prove correctness or execution.

Target state:
- The correct state required by the global task and the current group if the group is useful.
- Derive only instruction-grounded requirements from the Global Task Instruction and explicit task rules.
- The current subgoal name is a hypothesis from trajectory segmentation, not ground truth.
- Do not assume the current subgoal is a correct route to the global task.
- If the current subgoal follows a route that task rules already identify as wrong or unsupported, target_state must describe the correct-route target state, not "recognize/verify that the wrong route fails".
- Do not copy the agent's proposed answer from Thought/Action unless it is verified by the task rules.
- target_state must not contradict rules. If the agent's intended value conflicts with the rules, target_state must follow the rules.

Rules:
- Exact constraints needed to judge correctness: values, labels, mappings, thresholds, formulas, formatting, line breaks, separators, row/column/cell/field boundaries, ordering, and what must remain unchanged.
- Include task-level rules needed to judge whether the current subgoal is a correct, exploratory, or wrong route.
- For mapping/classification/threshold tasks, explicitly compute or state the rule that determines the expected label/value.

Preferred process:
- The correct or robust task-solving process when a scalable method or known UI/tool route is relevant.
- Do not require exact imitation when another process achieves the same correct target state.
- Do not describe the agent's attempted process as preferred merely because it appears in the trajectory.
- If the current group is a wrong-path or exploratory attempt, describe the correct route or state that the attempt is only diagnostic/exploratory.
- If task rules already identify a route, tool, menu, or operation as wrong or unsupported, preferred_process must start from the correct route instead of including that known-wrong attempt as a required confirmation step, unless the original user task explicitly asks to test that route. The agent's subgoal name, reasoning, or later failure does not count as a user request to test it.
- For spreadsheet/data tasks, prefer formulas, fill-down, range edits, sorting/filtering, and batch operations over repeated cell-by-cell edits.
"""

PHASE1_CONSTRAINTS = """
Phase 1: Step-Level Critic
- Purpose: score each target step using two axes: action execution and subgoal contribution.
- Evaluate each target step independently, then use later steps only to judge whether its effect was useful.
- For finished() actions, judge whether stopping/claiming completion was appropriate.

Intent summary:
- The immediate GUI action intended by the agent, inferred from Thought/Agent Response/Action text.
- The agent's stated rule or category is not authoritative; verify it against the task reference.

Expected subgoal effect:
- What this step should accomplish according to the task/subgoal rules, independent of the agent's Thought.
- Derive this from task_reference.rules and visible source data; do not copy the agent's unverified proposed answer.
- For mapping/classification/threshold/formula/text-format tasks, compute the expected value, label, line break, separator, cell/field/range, order, or format from the task reference.

Observed effect:
- The actual UI/state/content change visible in screenshots, or why the evidence is insufficient.

Execution result:
- Completed: the immediate GUI action executed as intended and caused the expected UI/state change.
- Failed: no change, wrong change, error state, repeated same intent, admitted failure, or premature finished().
- Uncertain: available visual/text evidence is insufficient.

Subgoal contribution:
- Useful: the effect is correct under the task/subgoal rules and directly advances the subgoal.
- Inefficient: the effect is correct and useful, but the action is manual, repetitive, brittle, or unnecessarily indirect.
- Exploratory: reasonable information-gathering or trial that helps choose the next action.
- Redundant: harmless repetition or low-value action that does not meaningfully advance the subgoal.
- Recovery: repairs or tries to repair a previous mistake.
- Harmful: wrong value, wrong label, wrong line break, wrong separator, wrong cell/field/range, wrong order, wrong format, wrong mapping/threshold/formula, corrupted target state, or moves away from the subgoal.

Axis relationship:
- execution_result compares intent_summary against observed_effect.
- subgoal_contribution compares expected_subgoal_effect against intent_summary and observed_effect.

Step score:
- 1.0: Completed + Useful, necessary, and efficient; or appropriate finished() after visible completion.
- 0.8: Completed + Useful with only minor inefficiency.
- 0.6: Completed + Inefficient but semantically correct and still useful.
- 0.4: Exploratory or partial progress.
- 0.2: Redundant/low-value but harmless.
- 0.0: Failed, Harmful, wrong target/content/format/rule application, or premature/false finished().

Step scoring constraints:
- Never mark execution_result as Completed from Thought alone; visual/state evidence must support it.
- If the next step repeats the same intent, the previous step is likely Failed or Uncertain.
- Topical relevance is not enough for Useful; the intended/observed effect must be correct under the task/subgoal constraints.
- If subgoal_contribution is Harmful, score must be 0.0 even if execution_result is Completed.
- If intent_summary or observed_effect conflicts with expected_subgoal_effect, subgoal_contribution must be Harmful and score must be 0.0, unless the step is explicitly Exploratory or Recovery.
"""

PHASE2_CONSTRAINTS = """
Phase 2: Current Group Conclusion
- Purpose: score the current subgoal group as a whole using two axes: subgoal status and goal contribution.
- Evaluate the group after reviewing all step-level judgments.

Subgoal intent summary:
- What this group is trying to accomplish, inferred from the subgoal name, reasoning, and actions.
- The subgoal name is not authoritative; summarize the attempted intent without assuming it is correct.

Expected goal effect:
- The effect this group should have on the global task after judging whether the current subgoal route is correct or wrong.
- If the route is correct, state the concrete task-relevant effect it should produce: correct page/menu/tool, correct file/content/value/layout, correct generated output, correct setting, correct selection, or correct progress toward the requested result.
- If the route is wrong, unsupported, unnecessary, or based on a false assumption, state that following this route is not expected to produce useful progress toward the global task.
- Do not convert a wrong route into a useful diagnostic objective unless the user explicitly asked to test or verify that route.

Observed group effect:
- The actual final UI/file/state/content effect of the group, based on the merged step-level judgments and their visual-evidence reasons.

Subgoal status:
- Completed: the target subgoal is visibly achieved by the result state.
- Partial: meaningful progress but not complete.
- Failed: no meaningful progress or final state contradicts the target.
- Do not mark Completed if the observed group effect does not satisfy the expected goal effect.

Goal contribution:
- Useful: the group follows a correct route under task_reference.rules/preferred_process and directly advances the global task by producing correct content/value/layout, opening the correct page/menu/tool, selecting the correct object/range/file, generating the correct output, applying the correct setting, or reaching the requested result.
- Exploratory: the route was genuinely uncertain before this group, and the group produced new information needed to choose the next action.
- Redundant: harmless repetition or low-value work that does not meaningfully improve global progress.
- Misaligned: wrong tool, wrong menu/location, wrong file/object, wrong timing, wrong constraints, unsupported operation, unnecessary route, or route based on a false assumption.
- Harmful: wrong content/value/layout, wrong generated output, corrupted required state, blocked progress, deleted/overwrote needed data, or reinforced a wrong assumption.
- A known-wrong route cannot be Useful merely because it was executed cleanly or confirmed an already-known failure.
- If subgoal_intent_summary or observed_group_effect conflicts with expected_goal_effect, goal_contribution should be Misaligned or Harmful, not Useful.

Group score:
- 1.0: Completed + Useful, with no meaningful harmful or wrong-content steps.
- 0.8: Completed + Useful, but with minor redundant/inefficient steps.
- 0.6: Partial + Useful, making meaningful progress toward the global task.
- 0.4: Exploratory with concrete diagnostic information, recovery, or useful information gain; or Partial with weak progress.
- 0.2: Redundant or Misaligned but with minor progress or weak diagnostic value.
- 0.0: Harmful, or Misaligned/Redundant with no useful information, regardless of local completion.

Group scoring constraints:
- A group cannot be Completed if key resulting content/value/layout violates the task reference.
- Do not let Previous Group Results overwrite current visual/step evidence; use them only as historical context.
"""

PHASE3_CONSTRAINTS = """
3. **Phase 3: Trajectory-Level Group Refinement**
   - Purpose: refine group-level scores after all groups have initial step-level and group-level judgments.
   - Refine only group-level conclusions; do not change step-level execution results.

   - Cross-group dependency:
       - Use earlier group results to judge whether a later group depends on valid or invalid prerequisites.
       - Use later group results to judge whether an earlier group was reused, invalidated, repaired, abandoned, or became a dead end.
       - Do not punish a correct group only because a later unrelated group failed.
   - Refined subgoal status:
       - Completed: the group result remains visibly/semantically achieved after considering cross-group evidence.
       - Partial: the group made meaningful progress but later evidence shows the result was incomplete, weakly used, or only partly valid.
       - Failed: the group produced no meaningful valid result, or later evidence shows its result was invalid.
   - Refined goal contribution:
       - Useful: the group result is correct under task rules and is reused by later groups or directly supports final task progress.
       - Exploratory: the group produced concrete diagnostic information that later helped choose a better route.
       - Redundant: the group was harmless but not meaningfully used or needed.
       - Misaligned: the group followed a wrong tool, wrong route, wrong object, wrong timing, unsupported operation, or false assumption.
       - Harmful: the group produced wrong content/value/output, corrupted required state, blocked progress, or caused later groups to build on invalid state.
   - Refined group score:
       - 1.0: Completed + Useful, and cross-group evidence confirms the result was valid and important.
       - 0.8: Completed + Useful, but with minor redundancy or limited downstream importance.
       - 0.6: Partial + Useful, with meaningful downstream use or valid progress.
       - 0.4: Exploratory with concrete downstream value, recovery, or weak partial progress.
       - 0.2: Redundant or Misaligned with minor progress or weak diagnostic value.
       - 0.0: Harmful, or Misaligned/Redundant with no useful downstream value.
   - Trajectory outcome reward:
       - Judge whether the complete trajectory satisfies the Global Task Instruction after considering all group results and dependencies.
       - Do not average group scores. Weight required end states, key deliverables, unresolved errors, successful corrections, and whether the final state is valid.
       - 1.0: All key task requirements are completed with reliable evidence.
       - 0.8: The task is essentially completed, with only a minor non-critical omission or uncertainty.
       - 0.6: Substantial correct completion, but at least one important requirement remains incomplete.
       - 0.4: Limited valid task progress; the main requested outcome is not completed.
       - 0.2: Minimal valid progress, with most required work missing or misaligned.
       - 0.0: No valid task completion, or the final state is harmful, invalid, or based on a false completion claim.
   - Group refinement constraints:
       - Do not hallucinate visual evidence beyond the step-level reasons and group conclusions.
       - Do not raise a harmful or wrong-content group only because the final outcome succeeded.
       - Do not lower a correct useful group only because a later unrelated group failed.
       - If a later group depends on an invalid output from an earlier group, reflect the invalid dependency in the later group's refined contribution.
"""

def clean_json_response(content: str) -> str:
    content = re.sub(r"```json\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```\s*", "", content)
    return content.strip()


async def _call_with_structure_retry(
    async_call_llm_fn, messages: List[Dict], max_tokens: int, validator, phase_name: str
):
    last_error = None
    for attempt in range(1, PRM_MAX_ATTEMPTS + 1):
        attempt_messages = messages + ([validation_feedback(last_error)] if last_error else [])
        response_content = await async_call_llm_fn(
            attempt_messages, max_tokens=max_tokens
        )
        try:
            if not response_content:
                raise PRMValidationError(f"{phase_name} response must not be empty")
            parsed_json = json.loads(clean_json_response(response_content))
            return validator(parsed_json)
        except (json.JSONDecodeError, PRMValidationError, TypeError, ValueError) as error:
            last_error = error
            print(
                f"[PRM Score] {phase_name} validation failed on attempt "
                f"{attempt}/{PRM_MAX_ATTEMPTS}: {error}"
            )
    return None

def encode_image_to_base64(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception:
        return None

def get_base_subgoal_name(name: str) -> str:
    return re.sub(r"\s*\((Part|Window)\s*\d+\)$", "", name).strip()

def _split_context_and_steps(raw_steps_data: List[Dict]) -> Tuple[Optional[Dict], List[Dict]]:
    if raw_steps_data and "[PREVIOUS CONTEXT]" in raw_steps_data[0].get('content', ''):
        return raw_steps_data[0], raw_steps_data[1:]
    return None, raw_steps_data

def _normalize_windows(current_sg_info: Dict) -> List[Dict]:
    windows = current_sg_info.get('windows_data') or []
    normalized = []
    for idx, window in enumerate(windows, start=1):
        steps_data = window.get('steps_data', []) or []
        if not steps_data:
            continue
        normalized.append({
            'window_idx': int(window.get('window_idx', idx)),
            'start_step': int(window.get('start_step', steps_data[0].get('step_idx', idx))),
            'end_step': int(window.get('end_step', steps_data[-1].get('step_idx', idx))),
            'steps_data': steps_data,
        })
    if normalized:
        return normalized

    context_step, pure_steps = _split_context_and_steps(current_sg_info.get('steps_data', []))
    if not pure_steps:
        return []

    for idx in range(0, len(pure_steps), MAX_STEPS_PER_WINDOW):
        chunk = pure_steps[idx:idx + MAX_STEPS_PER_WINDOW]
        if chunk:
            normalized.append({
                'window_idx': len(normalized) + 1,
                'start_step': chunk[0]['step_idx'],
                'end_step': chunk[-1]['step_idx'],
                'steps_data': chunk,
            })
    if normalized and context_step:
        normalized[0]['context_step'] = context_step
    return normalized

def _build_step_text(step: Dict) -> str:
    return f"Step {step.get('step_idx')}: {step.get('content', '')}"


def _format_group_step_text_context(steps: List[Dict]) -> str:
    lines = []
    seen = set()
    for step in steps:
        step_idx = step.get('step_idx')
        if step_idx is None or step_idx in seen:
            continue
        seen.add(step_idx)
        lines.append(_build_step_text(step))
    return "\n\n".join(lines) if lines else "N/A"


def _format_group_record(subgoal: str, reasoning: str, status: str = "N/A", score: str = "N/A", summary: str = "N/A", contribution: str = "N/A", group_label: str = "") -> str:
    prefix = f"Group {group_label}:\n" if group_label else ""
    return (
        f"{prefix}"
        f"  subgoal: {subgoal}\n"
        f"  reasoning: {reasoning}\n"
        f"  score: {score}\n"
        f"  summary: {summary}\n"
        f"  status: {status}\n"
        f"  contribution: {contribution}"
    )


def _format_task_reference(task_reference: Dict) -> str:
    if not isinstance(task_reference, dict) or not task_reference:
        return "N/A"
    return (
        f"  target_state: {task_reference.get('target_state', 'N/A')}\n"
        f"  rules: {task_reference.get('rules', 'N/A')}\n"
        f"  preferred_process: {task_reference.get('preferred_process', 'N/A')}"
    )
def _collect_future_group_records(all_subgoals: List[Dict], current_idx: int, current_base_name: str) -> str:
    records = []
    seen = set()
    for idx in range(current_idx + 1, len(all_subgoals)):
        item = all_subgoals[idx]
        base = get_base_subgoal_name(item.get("subgoal_name", ""))
        if not base or base == current_base_name or base in seen:
            continue
        seen.add(base)
        records.append(_format_group_record(base, item.get("reasoning", "N/A"), group_label=str(idx + 1)))
    return "\n".join(records) if records else "N/A (End of Task)"

async def async_score_recursive_step_multi_turn(
    task_instruction: str,
    task_root_path: str,
    all_subgoals: List[Dict],
    current_idx: int,
    previous_group_results: str,
    async_call_llm_fn
) -> Optional[Dict]:
    current_sg_info = all_subgoals[current_idx]
    raw_subgoal_name = current_sg_info.get('subgoal_name', 'Unknown')
    current_base_name = get_base_subgoal_name(raw_subgoal_name)
    current_reasoning = current_sg_info.get('reasoning', 'N/A')

    current_group_record = _format_group_record(current_base_name, current_reasoning, status='N/A', score='N/A', summary='Current target group', group_label=str(current_idx + 1))
    future_group_records = _collect_future_group_records(all_subgoals, current_idx, current_base_name)

    windows = _normalize_windows(current_sg_info)
    if not windows:
        return None

    ordered_windows = sorted(windows, key=lambda w: (w.get("start_step", 0), w.get("window_idx", 0)))
    ordered_steps = []
    for window in ordered_windows:
        for step in window.get("steps_data", []) or []:
            if step.get("step_idx") is not None:
                ordered_steps.append(step)
    ordered_steps.sort(key=lambda s: s.get("step_idx", 0))
    group_step_text_context = _format_group_step_text_context(ordered_steps)

    system_prompt_p0 = """
    You are an expert Visual Reward Model for GUI Agents.
    Build a rule-consistent task reference for later judgments.
    
    {core_scoring_prompt}

    """.format(core_scoring_prompt=CORE_SCORING_PROMPT)

    user_content_p0 = [{
        'type': 'text',
        'text': f"""
## Global Task Instruction
{task_instruction}

## Subgoal Trace (Context):
**1. History Group Records:**
{previous_group_results if previous_group_results else 'N/A (No previous scored groups)'}

**2. Current Group Record:**
{current_group_record}

**3. Future Group Records:**
{future_group_records}

## Current Group Step Text (Reference Only)
{group_step_text_context}

## Audit Instructions for PHASE 0
{PHASE0_CONSTRAINTS}

## Output Format (JSON) for PHASE 0
{{
  "task_reference": {{
    "target_state": "<string> Exact target state required by the task/subgoal.",
    "rules": "<string> Values, mappings, thresholds, formatting, boundaries, ordering, formulas, or constraints that must be checked.",
    "preferred_process": "<string> Expected high-quality/scalable process when relevant."
  }}
}}

Hard output constraints:
- Return valid JSON only, with no markdown fences.
"""
    }]

    conversation_history_p0 = [
        {'role': 'system', 'content': system_prompt_p0},
        {'role': 'user', 'content': user_content_p0},
    ]

    task_reference = await _call_with_structure_retry(
        async_call_llm_fn,
        conversation_history_p0,
        2048,
        validate_task_reference,
        "Phase 0",
    )
    if task_reference is None:
        return None
    task_reference_text = _format_task_reference(task_reference)

    merged_step_critics = []

    for window in ordered_windows:
        current_window_steps = window.get('steps_data', [])
        if not current_window_steps:
            continue

        target_step_indices = [s.get('step_idx') for s in current_window_steps if s.get('step_idx') is not None]
        if not target_step_indices:
            continue

        current_step_indices = {int(s.get('step_idx')) for s in current_window_steps if s.get('step_idx') is not None}
        current_last_step_idx = max(current_step_indices)
        current_final_image_name = current_window_steps[-1].get('next_image', '')

        system_prompt_p1 = """
        You are an expert Visual Reward Model for GUI Agents.
        Score the target steps of the current group.
        
        {core_scoring_prompt}

        """.format(core_scoring_prompt=CORE_SCORING_PROMPT)

        user_content_p1 = [{
            'type': 'text',
            'text': f"""
## Global Task Instruction
{task_instruction}

## Task Reference:
{task_reference_text}

## Target Subgoal:
Subgoal: {current_base_name}
Reasoning: {current_reasoning}

## Audit Instructions for PHASE 1
{PHASE1_CONSTRAINTS}

## Current Group Trace:
"""
        }]

        for step in ordered_steps:
            step_idx = step.get('step_idx')
            if step_idx is None:
                continue
            img_path = os.path.join(task_root_path, step.get('image', ''))
            if int(step_idx) in current_step_indices:
                user_content_p1.append({'type': 'text', 'text': "\n[Target Step Evidence]:"})
                b64 = encode_image_to_base64(img_path)
                if b64:
                    user_content_p1.append({'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}})
                user_content_p1.append({'type': 'text', 'text': _build_step_text(step)})
                if int(step_idx) == current_last_step_idx and current_final_image_name:
                    final_image_path = os.path.join(task_root_path, current_final_image_name)
                    b64_end = encode_image_to_base64(final_image_path)
                    if b64_end:
                        user_content_p1.append({'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64_end}'}})
            else:
                user_content_p1.append({'type': 'text', 'text': f"\n[Context Text Only]: {_build_step_text(step)}"})

        user_content_p1.append({
            'type': 'text',
            'text': f"""
## Output Format (JSON) for PHASE 1
{{
  "step_critic": [
    {{
      "step_index": <int> (MUST be one of: [{', '.join(map(str, target_step_indices))}]),
      "is_finished_action": <bool>,
      "intent_summary": "<string> Intended action inferred from text/action.",
      "expected_subgoal_effect": "<string> What this step should accomplish according to the task/subgoal rules, independent of the agent's Thought.",
      "observed_effect": "<string> Actual UI/state change observed from screenshots, or why uncertain.",
      "execution_result": "<string> 'Completed', 'Failed', or 'Uncertain'.",
      "subgoal_contribution": "<string> 'Useful', 'Inefficient', 'Exploratory', 'Redundant', 'Recovery', or 'Harmful'.",
      "score": <float> (MUST be one of: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
      "reason": "<string>"
    }}
  ]
}}

Hard output constraints:
- Return valid JSON only, with no markdown fences.
- step_critic must contain exactly one object for every target step index: [{', '.join(map(str, target_step_indices))}].
- Do not include context-only steps.
- Preserve ascending step_index order.
- Reasons for standard actions must mention intended action text, expected_subgoal_effect, observed visual/state evidence, and whether the observed content/value/layout satisfies the task/subgoal rules.
"""
        })

        conversation_history_p1 = [
            {'role': 'system', 'content': system_prompt_p1},
            {'role': 'user', 'content': user_content_p1},
        ]

        phase1_critics = await _call_with_structure_retry(
            async_call_llm_fn,
            conversation_history_p1,
            4096,
            lambda result, indices=target_step_indices: validate_step_critics(result, indices),
            f"Phase 1 window {window.get('window_idx', 'N/A')}",
        )
        if phase1_critics is None:
            return None
        merged_step_critics.extend(phase1_critics)

    expected_group_indices = [int(step['step_idx']) for step in ordered_steps]
    merged_step_critics = validate_step_critics(
        {'step_critic': merged_step_critics}, expected_group_indices
    )
    current_group_evidence = '\n'.join(
        f"- Step {item.get('step_index')}: score={item.get('score', 'N/A')}, reason={item.get('reason', 'N/A')}"
        for item in merged_step_critics
    )

    system_prompt_p2 = """
    You are an expert Visual Reward Model for GUI Agents.
    Your goal is to evaluate one target Subgoal Group and assign forward process rewards.

    {core_scoring_prompt}

    """.format(core_scoring_prompt=CORE_SCORING_PROMPT)

    user_content_p2 = [{
        'type': 'text',
        'text': f"""
## Global Task Instruction
{task_instruction}

## Task Reference:
{task_reference_text}

## Subgoal Trace (Context):
**1. History Group Records:**
{previous_group_results if previous_group_results else 'N/A (No previous scored groups)'}

**2. Current Group Record:**
{current_group_record}

**3. Future Group Records:**
{future_group_records}

## Current Group Step-Level Evidence:
{current_group_evidence if current_group_evidence else 'N/A'}

## Audit Instructions for PHASE 2
{PHASE2_CONSTRAINTS}

## Output Format (JSON) for PHASE 2
{{
  "current_group_conclusion": {{
    "subgoal_intent_summary": "<string> What this group is trying to accomplish, without assuming the subgoal is correct.",
    "expected_goal_effect": "<string> What this group should contribute to the global task if it is correct/useful.",
    "observed_group_effect": "<string> Actual final UI/file/state/content effect of the group.",
    "subgoal_status": "<string> 'Completed', 'Partial', or 'Failed'.",
    "goal_contribution": "<string> 'Useful', 'Exploratory', 'Redundant', 'Misaligned', or 'Harmful'.",
    "score": <float> (MUST be one of: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "reason": "<string>",
    "history_summary_for_next_group": "<string> One concise sentence stating what was achieved, failed, or learned for later groups."
  }}
}}

Hard output constraints:
- Return valid JSON only, with no markdown fences.
- The group conclusion must be based on the full set of step_critic evidence, not only the best or last step.
- A group cannot receive a high score if its key steps are harmful, wrong-content, or built on a route contradicted by task_reference.
"""
    }]

    conversation_history_p2 = [
        {'role': 'system', 'content': system_prompt_p2},
        {'role': 'user', 'content': user_content_p2},
    ]

    group_conclusion = await _call_with_structure_retry(
        async_call_llm_fn,
        conversation_history_p2,
        2048,
        validate_group_conclusion,
        "Phase 2",
    )
    if group_conclusion is None:
        return None

    final_result = {
        'task_reference': task_reference,
        'step_critic': merged_step_critics,
        'current_group_conclusion': group_conclusion
    }
    return final_result


def _format_phase3_group_evaluations(all_subgoals: List[Dict]) -> str:
    records = []
    for idx, item in enumerate(all_subgoals):
        eval_res = item.get("evaluation")
        if not isinstance(eval_res, dict):
            continue
        task_reference = eval_res.get("task_reference", {})
        group_conclusion = eval_res.get("current_group_conclusion", {})
        step_lines = []
        for sc in eval_res.get("step_critic", []):
            if not isinstance(sc, dict):
                continue
            step_lines.append(
                f"    Step {sc.get('step_index', 'N/A')}: "
                f"score={sc.get('score', 'N/A')}, "
                f"execution={sc.get('execution_result', 'N/A')}, "
                f"contribution={sc.get('subgoal_contribution', 'N/A')}, "
                f"reason={sc.get('reason', 'N/A')}"
            )
        records.append(
            f"Group {idx + 1}:\n"
            f"  subgoal: {item.get('subgoal_name', 'Unknown')}\n"
            f"  reasoning: {item.get('reasoning', 'N/A')}\n"
            f"  task_reference:\n"
            f"    target_state: {task_reference.get('target_state', 'N/A') if isinstance(task_reference, dict) else 'N/A'}\n"
            f"    rules: {task_reference.get('rules', 'N/A') if isinstance(task_reference, dict) else 'N/A'}\n"
            f"    preferred_process: {task_reference.get('preferred_process', 'N/A') if isinstance(task_reference, dict) else 'N/A'}\n"
            f"  step_level_evidence:\n"
            f"{chr(10).join(step_lines) if step_lines else '    N/A'}\n"
            f"  initial_group_conclusion:\n"
            f"    subgoal_status: {group_conclusion.get('subgoal_status', 'N/A')}\n"
            f"    goal_contribution: {group_conclusion.get('goal_contribution', 'N/A')}\n"
            f"    score: {group_conclusion.get('score', 'N/A')}\n"
            f"    reason: {group_conclusion.get('reason', 'N/A')}\n"
            f"    history_summary_for_next_group: {group_conclusion.get('history_summary_for_next_group', 'N/A')}"
        )
    return "\n\n".join(records) if records else "N/A"


async def async_refine_group_scores_multi_turn(
    task_instruction: str,
    all_subgoals: List[Dict],
    async_call_llm_fn
) -> Optional[Dict]:
    group_evaluations = _format_phase3_group_evaluations(all_subgoals)
    scored_group_indices = [
        idx + 1 for idx, item in enumerate(all_subgoals)
        if isinstance(item.get("evaluation"), dict)
    ]
    target_group_indices_str = ", ".join(map(str, scored_group_indices))
    if not scored_group_indices:
        return None

    system_prompt = """
    You are an expert Visual Reward Model for GUI Agents.
    Your goal is to refine group-level process rewards using full trajectory-level group evidence.

    Core scoring principle:
    - Initial group conclusions provide local group judgments; cross-group evidence can validate, weaken, or invalidate them.
    - Infer the trajectory outcome reward from the complete group evidence.

    """

    user_content = [{
        "type": "text",
        "text": f"""
## Global Task Instruction
{task_instruction}

## Initial Group-Level Evaluations
{group_evaluations}

## Audit Instructions for PHASE 3
{PHASE3_CONSTRAINTS}

## Output Format (JSON) for PHASE 3
{{
  "refined_group_conclusions": [
    {{
      "group_index": <int> (MUST be one of: [{target_group_indices_str}]),
      "refined_subgoal_status": "<string> 'Completed', 'Partial', or 'Failed'.",
      "refined_goal_contribution": "<string> 'Useful', 'Exploratory', 'Redundant', 'Misaligned', or 'Harmful'.",
      "refined_score": <float> (MUST be one of: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
      "reason": "<string>"
    }}
  ],
  "trajectory_outcome": {{
    "reward": <float> (MUST be one of: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "reason": "<string> Concise synthesis of final task completion across all groups."
  }}
}}

Hard output constraints:
- Return valid JSON only, with no markdown fences.
- Include exactly one object for every scored group index: [{target_group_indices_str}].
- Preserve ascending group_index order.
- Each refined group object must contain exactly the listed fields.
- If no refinement is needed, copy the original status, contribution, and score into the refined fields.
"""
    }]

    original_conclusions = collect_original_group_conclusions(all_subgoals)
    return await _call_with_structure_retry(
        async_call_llm_fn,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        4096,
        lambda result: validate_phase3_refinement(result, original_conclusions),
        "Phase 3",
    )
