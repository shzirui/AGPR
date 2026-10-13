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
- Evaluate from task rules, screenshots, actions, and previous group results.
- Step axes judge local execution and subgoal contribution; group axes judge subgoal status and goal contribution.
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
- Purpose: evaluate each target step using two axes: action execution and subgoal contribution.
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
- This is the low-level grounding axis. Compare intent_summary with observed_effect; do not judge whether the semantic purpose was correct here.
- Completed: the intended immediate GUI effect is fully supported by visual/state evidence, regardless of whether that intent was semantically correct or efficient; Thought alone is insufficient.
- Partial: only part of the intended immediate GUI effect is visibly achieved.
- Failed: no change, wrong change, error state, repeated same intent, admitted failure, or premature finished().
- Uncertain: visual/text evidence is insufficient to determine execution; use it only for grounding uncertainty, not uncertainty about semantic contribution.

Subgoal contribution:
- This is the high-level plan/strategy axis. Compare intent_summary with expected_subgoal_effect and task_reference to judge the purpose, method, and efficiency, separately from whether the GUI action executed.
- Milestone: the intended effect is the decisive operation that would complete the current subgoal or produce its required deliverable/end state, including a successful correction that directly establishes that state; preparation and navigation are not milestones.
- UsefulEfficient: the semantic purpose is correct under the task/subgoal constraints, not merely topically related, and usefully advances the subgoal through an appropriate, direct, and scalable method; this includes a necessary on-path transition, correct decision, or successful recovery/error correction.
- UsefulInefficient: the semantic purpose is correct and useful, including a useful transition or recovery, but the method is substantially manual, repetitive, brittle, indirect, or non-scalable; for example, filling cells individually when a reliable batch fill is available.
- Exploratory: the semantic purpose is reasonable information-gathering needed under genuine uncertainty, even when it is not direct task progress.
- Redundant: the semantic purpose is unnecessary, repetitive, meaningless, or off-path and provides neither useful progress nor useful information.
- Harmful: the semantic purpose is an actively wrong or harmful off-path action, false completion claim, ignored needed correction, or otherwise targets a wrong value, label, format, object, route, rule, or state.

Cross-axis rules:
- Judge the two axes independently. A semantically wrong action can be fully executed; use Completed + Harmful in that case.
- Partial means that the step did not fully realize its own immediate intended effect, not that the entire subgoal remains incomplete.
"""

PHASE2_CONSTRAINTS = """
Phase 2: Current Group Conclusion
- Purpose: evaluate the current subgoal group using two axes: subgoal status and goal contribution.
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
- This is the group-level grounding axis. Compare subgoal_intent_summary with observed_group_effect, independently of whether the group intent was semantically correct.
- Completed: the group fully realized its intended group-level effect.
- Partial: the group realized only part of its intended group-level effect.
- Failed: the group did not realize its intended group-level effect.
- Uncertain: the available step/visual evidence is insufficient to determine group-level execution; do not use it for uncertainty about semantic contribution.

Goal contribution:
- This is the global plan/strategy axis. Compare subgoal_intent_summary with expected_goal_effect and task_reference to judge the route, method, and efficiency, separately from whether the group intent was realized.
- Milestone: the intended group effect is decisive for completing an important global-task requirement or producing its required deliverable/end state, including a successful correction that directly establishes that state; preparation and navigation groups are not milestones.
- UsefulEfficient: the group intent is correct under the task constraints, not merely topically related, and advances the global task through an appropriate, direct, and scalable route; this includes a necessary on-path transition, correct decision, or successful recovery/error correction.
- UsefulInefficient: the group intent is correct and useful, including a useful transition or recovery, but its route is substantially manual, repetitive, brittle, indirect, or non-scalable.
- Exploratory: the group reasonably gathers information needed under genuine uncertainty, even when it does not directly advance the task.
- Redundant: the group is unnecessary, repetitive, meaningless, or off-path and provides neither useful progress nor useful information.
- Harmful: the group follows an actively wrong or harmful route, makes a false completion claim, ignores a needed correction, produces wrong content/output, corrupts required state, or otherwise moves away from the global task.

Cross-axis rules:
- Judge the two axes independently. A semantically wrong group can fully realize its intended effect; use Completed + Harmful in that case.
- Partial means that the group only partly realized its own intended group-level effect, not that the entire global task remains incomplete.
- Base observed_group_effect on current step evidence; Previous Group Results are historical context and do not override it.
"""

PHASE3_CONSTRAINTS = """
3. **Phase 3: Trajectory-Level Group-Judgment Refinement**
   - Purpose: refine group-level judgments after all groups have initial step-level and group-level judgments.
   - Refine only group-level conclusions; do not change step-level execution results.

   - Cross-group dependency:
       - Use earlier group results to judge whether a later group depends on valid or invalid prerequisites.
       - Use later group results to judge whether an earlier group was reused, invalidated, repaired, abandoned, or became a dead end.
       - Do not punish a correct group only because a later unrelated group failed.
   - Refined subgoal status:
       - Completed: the group result remains visibly/semantically achieved after considering cross-group evidence.
       - Partial: the group made meaningful progress but later evidence shows the result was incomplete, weakly used, or only partly valid.
       - Failed: the group produced no meaningful valid result, or later evidence shows its result was invalid.
       - Uncertain: cross-group evidence is still insufficient to determine whether the intended group effect was realized.
   - Refined goal contribution:
       - Milestone: the group is decisive for a major task requirement or required deliverable/end state.
       - UsefulEfficient: the group follows a correct, direct, and scalable route that supports global task progress.
       - UsefulInefficient: the group supports global task progress through a correct but substantially inefficient route.
       - Exploratory: the group produced concrete diagnostic information that later helped choose a better route.
       - Redundant: the group was harmless but not meaningfully used or needed.
       - Harmful: the group followed a wrong route, produced wrong content/value/output, corrupted required state, blocked progress, or caused later groups to build on invalid state.
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
      "execution_result": "<string> 'Completed', 'Partial', 'Failed', or 'Uncertain'.",
      "subgoal_contribution": "<string> 'Milestone', 'UsefulEfficient', 'UsefulInefficient', 'Exploratory', 'Redundant', or 'Harmful'.",
      "reason": "<string>"
    }}
  ]
}}

Hard output constraints:
- Return valid JSON only, with no markdown fences.
- step_critic must contain exactly one object for every target step index: [{', '.join(map(str, target_step_indices))}].
- Do not include context-only steps.
- Preserve ascending step_index order.
- Each step_critic object must contain exactly the listed fields.
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
        {'step_critic': merged_step_critics},
        expected_group_indices,
        allow_derived_score=True,
    )
    current_group_evidence = json.dumps([
        {
            field: item.get(field)
            for field in (
                "step_index",
                "intent_summary",
                "expected_subgoal_effect",
                "observed_effect",
                "execution_result",
                "subgoal_contribution",
                "reason",
            )
        }
        for item in merged_step_critics
    ], ensure_ascii=False, indent=2)

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
    "subgoal_status": "<string> 'Completed', 'Partial', 'Failed', or 'Uncertain'.",
    "goal_contribution": "<string> 'Milestone', 'UsefulEfficient', 'UsefulInefficient', 'Exploratory', 'Redundant', or 'Harmful'.",
    "reason": "<string>",
    "history_summary_for_next_group": "<string> One concise sentence stating what was achieved, failed, or learned for later groups."
  }}
}}

Hard output constraints:
- Return valid JSON only, with no markdown fences.
- The group conclusion must be based on the full set of step_critic evidence, not only the best or last step.
- current_group_conclusion must contain exactly the listed fields.
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
            f"    reason: {group_conclusion.get('reason', 'N/A')}\n"
            f"    history_summary_for_next_group: {group_conclusion.get('history_summary_for_next_group', 'N/A')}"
        )
    return "\n\n".join(records) if records else "N/A"


async def async_refine_group_judgments_multi_turn(
    task_instruction: str,
    all_subgoals: List[Dict],
    outcome_reward: float,
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
    Your goal is to refine group-level judgments using full trajectory-level group evidence.

    Core refinement principle:
    - Initial group conclusions provide local group judgments; cross-group evidence can validate, weaken, or invalidate them.
    - Final outcome reward is context, not automatic proof that every group was useful or harmful.

    """

    user_content = [{
        "type": "text",
        "text": f"""
## Global Task Instruction
{task_instruction}

## Final Outcome Reward
{outcome_reward}

## Initial Group-Level Evaluations
{group_evaluations}

## Audit Instructions for PHASE 3
{PHASE3_CONSTRAINTS}

## Output Format (JSON) for PHASE 3
{{
  "refined_group_conclusions": [
    {{
      "group_index": <int> (MUST be one of: [{target_group_indices_str}]),
      "refined_subgoal_status": "<string> 'Completed', 'Partial', 'Failed', or 'Uncertain'.",
      "refined_goal_contribution": "<string> 'Milestone', 'UsefulEfficient', 'UsefulInefficient', 'Exploratory', 'Redundant', or 'Harmful'.",
      "reason": "<string>"
    }}
  ]
}}

Hard output constraints:
- Return valid JSON only, with no markdown fences.
- Include exactly one object for every evaluated group index: [{target_group_indices_str}].
- Preserve ascending group_index order.
- Each object must contain exactly the listed fields.
- If no refinement is needed, copy the original status and contribution into the refined fields.
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
