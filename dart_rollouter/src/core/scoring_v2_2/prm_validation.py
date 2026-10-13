from typing import Dict, Iterable, List, Mapping, Optional, Sequence


PRM_MAX_ATTEMPTS = 3
VALID_SCORES = frozenset({0.0, 0.2, 0.4, 0.6, 0.8, 1.0})
VALID_EXECUTION_RESULTS = frozenset({"Completed", "Failed", "Uncertain"})
VALID_STEP_CONTRIBUTIONS = frozenset({
    "Useful", "Inefficient", "Exploratory", "Redundant", "Recovery", "Harmful",
})
VALID_GROUP_STATUSES = frozenset({"Completed", "Partial", "Failed"})
VALID_GROUP_CONTRIBUTIONS = frozenset({
    "Useful", "Exploratory", "Redundant", "Misaligned", "Harmful",
})


class PRMValidationError(ValueError):
    """Raised when a model response violates the deterministic output contract."""


def _object(value: object, path: str) -> Dict:
    if not isinstance(value, dict):
        raise PRMValidationError(f"{path} must be an object")
    return value


def _list(value: object, path: str) -> List:
    if not isinstance(value, list):
        raise PRMValidationError(f"{path} must be a list")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PRMValidationError(f"{path} must be a non-empty string")
    return value.strip()


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PRMValidationError(f"{path} must be an integer")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise PRMValidationError(f"{path} must be a boolean")
    return value


def _enum(value: object, allowed: Iterable[str], path: str) -> str:
    value = _string(value, path)
    if value not in allowed:
        raise PRMValidationError(f"{path}={value!r} is not a valid enum value")
    return value


def _score(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PRMValidationError(f"{path} must be a numeric score")
    value = float(value)
    if value not in VALID_SCORES:
        raise PRMValidationError(
            f"{path}={value!r} must be one of {sorted(VALID_SCORES)}"
        )
    return value


def validation_feedback(error: Exception) -> Dict:
    return {
        "role": "user",
        "content": [{
            "type": "text",
            "text": (
                "Your previous output failed schema validation: "
                f"{error}. Return a complete corrected JSON response only."
            ),
        }],
    }


def validate_grouping(raw_groups: object, total_steps: int) -> List[Dict]:
    total_steps = _integer(total_steps, "total_steps")
    if total_steps <= 0:
        raise PRMValidationError("total_steps must be positive")
    groups = _list(raw_groups, "grouping")
    if not groups:
        raise PRMValidationError("grouping must not be empty")

    normalized = []
    covered = []
    previous_end = 0
    for position, raw in enumerate(groups):
        path = f"grouping[{position}]"
        item = _object(raw, path)
        start = _integer(item.get("start_step"), f"{path}.start_step")
        end = _integer(item.get("end_step"), f"{path}.end_step")
        if not 1 <= start <= end <= total_steps:
            raise PRMValidationError(
                f"{path} range [{start}, {end}] must satisfy "
                f"1 <= start_step <= end_step <= {total_steps}"
            )
        if start != previous_end + 1:
            raise PRMValidationError(
                f"{path}.start_step must be {previous_end + 1}, got {start}"
            )
        copied = dict(item)
        copied["subgoal_name"] = _string(item.get("subgoal_name"), f"{path}.subgoal_name")
        copied["reasoning"] = _string(item.get("reasoning"), f"{path}.reasoning")
        copied["start_step"] = start
        copied["end_step"] = end
        normalized.append(copied)
        covered.extend(range(start, end + 1))
        previous_end = end

    expected = list(range(1, total_steps + 1))
    if covered != expected:
        raise PRMValidationError(
            f"grouping must cover steps exactly once in order: expected {expected}, got {covered}"
        )
    return normalized


def validate_grouped_step_coverage(
    subgoals: object,
    total_steps: int,
    require_windows: bool = False,
) -> None:
    groups = _list(subgoals, "subgoals")
    all_indices = []
    for position, raw in enumerate(groups):
        path = f"subgoals[{position}]"
        item = _object(raw, path)
        start = _integer(item.get("start_step"), f"{path}.start_step")
        end = _integer(item.get("end_step"), f"{path}.end_step")
        steps_data = _list(item.get("steps_data"), f"{path}.steps_data")
        pure_indices = []
        for step in steps_data:
            step = _object(step, f"{path}.steps_data[]")
            if "[PREVIOUS CONTEXT]" in str(step.get("content", "")):
                continue
            pure_indices.append(_integer(step.get("step_idx"), f"{path}.steps_data[].step_idx"))
        expected_group = list(range(start, end + 1))
        if pure_indices != expected_group:
            raise PRMValidationError(
                f"{path}.steps_data must cover {expected_group}, got {pure_indices}"
            )
        all_indices.extend(pure_indices)

        if require_windows:
            windows = _list(item.get("windows_data"), f"{path}.windows_data")
            if not windows:
                raise PRMValidationError(f"{path}.windows_data must not be empty")
            window_indices = []
            for window_position, raw_window in enumerate(windows):
                window_path = f"{path}.windows_data[{window_position}]"
                window = _object(raw_window, window_path)
                window_steps = _list(window.get("steps_data"), f"{window_path}.steps_data")
                indices = [
                    _integer(_object(step, f"{window_path}.steps_data[]").get("step_idx"),
                             f"{window_path}.steps_data[].step_idx")
                    for step in window_steps
                ]
                if not indices:
                    raise PRMValidationError(f"{window_path}.steps_data must not be empty")
                if indices != list(range(indices[0], indices[-1] + 1)):
                    raise PRMValidationError(f"{window_path} steps must be contiguous")
                if _integer(window.get("start_step"), f"{window_path}.start_step") != indices[0]:
                    raise PRMValidationError(f"{window_path}.start_step does not match its steps")
                if _integer(window.get("end_step"), f"{window_path}.end_step") != indices[-1]:
                    raise PRMValidationError(f"{window_path}.end_step does not match its steps")
                window_indices.extend(indices)
            if window_indices != expected_group:
                raise PRMValidationError(
                    f"{path}.windows_data must cover {expected_group}, got {window_indices}"
                )

    expected_all = list(range(1, total_steps + 1))
    if all_indices != expected_all:
        raise PRMValidationError(
            f"subgoals must cover steps exactly once: expected {expected_all}, got {all_indices}"
        )


def validate_task_reference(raw_result: object) -> Dict:
    result = _object(raw_result, "phase0")
    reference = _object(result.get("task_reference"), "phase0.task_reference")
    return {
        "target_state": _string(reference.get("target_state"), "phase0.task_reference.target_state"),
        "rules": _string(reference.get("rules"), "phase0.task_reference.rules"),
        "preferred_process": _string(
            reference.get("preferred_process"), "phase0.task_reference.preferred_process"
        ),
    }


def validate_step_critics(raw_result: object, expected_step_indices: Sequence[int]) -> List[Dict]:
    result = _object(raw_result, "phase1")
    items = _list(result.get("step_critic"), "phase1.step_critic")
    expected = list(expected_step_indices)
    if len(items) != len(expected):
        raise PRMValidationError(
            f"phase1.step_critic count mismatch: expected {len(expected)}, got {len(items)}"
        )

    normalized = []
    actual = []
    text_fields = ("intent_summary", "expected_subgoal_effect", "observed_effect", "reason")
    for position, raw in enumerate(items):
        path = f"phase1.step_critic[{position}]"
        item = _object(raw, path)
        copied = dict(item)
        copied["step_index"] = _integer(item.get("step_index"), f"{path}.step_index")
        copied["is_finished_action"] = _boolean(
            item.get("is_finished_action"), f"{path}.is_finished_action"
        )
        for field in text_fields:
            copied[field] = _string(item.get(field), f"{path}.{field}")
        copied["execution_result"] = _enum(
            item.get("execution_result"), VALID_EXECUTION_RESULTS, f"{path}.execution_result"
        )
        copied["subgoal_contribution"] = _enum(
            item.get("subgoal_contribution"),
            VALID_STEP_CONTRIBUTIONS,
            f"{path}.subgoal_contribution",
        )
        copied["score"] = _score(item.get("score"), f"{path}.score")
        actual.append(copied["step_index"])
        normalized.append(copied)

    if actual != expected:
        raise PRMValidationError(
            f"phase1.step_critic indices must exactly equal {expected}, got {actual}"
        )
    return normalized


def validate_group_conclusion(raw_result: object) -> Dict:
    result = _object(raw_result, "phase2")
    conclusion = _object(
        result.get("current_group_conclusion"), "phase2.current_group_conclusion"
    )
    normalized = dict(conclusion)
    for field in (
        "subgoal_intent_summary",
        "expected_goal_effect",
        "observed_group_effect",
        "reason",
        "history_summary_for_next_group",
    ):
        normalized[field] = _string(conclusion.get(field), f"phase2.current_group_conclusion.{field}")
    normalized["subgoal_status"] = _enum(
        conclusion.get("subgoal_status"),
        VALID_GROUP_STATUSES,
        "phase2.current_group_conclusion.subgoal_status",
    )
    normalized["goal_contribution"] = _enum(
        conclusion.get("goal_contribution"),
        VALID_GROUP_CONTRIBUTIONS,
        "phase2.current_group_conclusion.goal_contribution",
    )
    normalized["score"] = _score(
        conclusion.get("score"), "phase2.current_group_conclusion.score"
    )
    return normalized


def validate_single_group_result(raw_result: object, expected_step_indices: Sequence[int]) -> Dict:
    result = _object(raw_result, "single_group_result")
    return {
        "task_reference": validate_task_reference(result),
        "step_critic": validate_step_critics(result, expected_step_indices),
        "current_group_conclusion": validate_group_conclusion(result),
    }


def validate_phase3_refinement(
    raw_result: object,
    original_conclusions: Mapping[int, Dict],
) -> Dict:
    result = _object(raw_result, "phase3")
    trajectory_outcome = _object(
        result.get("trajectory_outcome"), "phase3.trajectory_outcome"
    )
    normalized_outcome = {
        "reward": _score(
            trajectory_outcome.get("reward"),
            "phase3.trajectory_outcome.reward",
        ),
        "reason": _string(
            trajectory_outcome.get("reason"),
            "phase3.trajectory_outcome.reason",
        ),
    }
    items = _list(result.get("refined_group_conclusions"), "phase3.refined_group_conclusions")
    expected = sorted(original_conclusions)
    if len(items) != len(expected):
        raise PRMValidationError(
            f"phase3 group count mismatch: expected {len(expected)}, got {len(items)}"
        )

    normalized = []
    actual = []
    for position, raw in enumerate(items):
        path = f"phase3.refined_group_conclusions[{position}]"
        item = _object(raw, path)
        expected_fields = {
            "group_index",
            "refined_subgoal_status",
            "refined_goal_contribution",
            "refined_score",
            "reason",
        }
        actual_fields = set(item)
        if actual_fields != expected_fields:
            raise PRMValidationError(
                f"{path} fields must exactly equal {sorted(expected_fields)}, "
                f"got {sorted(actual_fields)}"
            )
        group_index = _integer(item.get("group_index"), f"{path}.group_index")
        original = original_conclusions.get(group_index)
        if original is None:
            raise PRMValidationError(f"{path}.group_index={group_index} is not a scored group")
        refined_subgoal_status = _enum(
            item.get("refined_subgoal_status"), VALID_GROUP_STATUSES,
            f"{path}.refined_subgoal_status",
        )
        refined_goal_contribution = _enum(
            item.get("refined_goal_contribution"), VALID_GROUP_CONTRIBUTIONS,
            f"{path}.refined_goal_contribution",
        )
        copied = {
            "group_index": group_index,
            "refined_subgoal_status": refined_subgoal_status,
            "refined_goal_contribution": refined_goal_contribution,
            "refined_score": _score(
                item.get("refined_score"), f"{path}.refined_score"
            ),
            "reason": _string(item.get("reason"), f"{path}.reason"),
        }
        actual.append(group_index)
        normalized.append(copied)

    if actual != expected:
        raise PRMValidationError(
            f"phase3 group indices must exactly equal {expected}, got {actual}"
        )
    return {
        "refined_group_conclusions": normalized,
        "trajectory_outcome": normalized_outcome,
    }


def collect_original_group_conclusions(all_subgoals: Sequence[Dict]) -> Dict[int, Dict]:
    conclusions = {}
    for index, subgoal in enumerate(all_subgoals, start=1):
        evaluation = subgoal.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        conclusion = evaluation.get("current_group_conclusion")
        if isinstance(conclusion, dict):
            conclusions[index] = conclusion
    return conclusions


def validate_final_step_coverage(subgoals: object, total_steps: int) -> None:
    groups = _list(subgoals, "subgoals")
    indices = []
    for group_position, raw_group in enumerate(groups):
        group = _object(raw_group, f"subgoals[{group_position}]")
        evaluation = _object(group.get("evaluation"), f"subgoals[{group_position}].evaluation")
        validate_task_reference(evaluation)
        validate_group_conclusion(evaluation)
        critics = _list(evaluation.get("step_critic"), f"subgoals[{group_position}].evaluation.step_critic")
        group_indices = [
            _integer(_object(item, "step_critic[]").get("step_index"), "step_critic[].step_index")
            for item in critics
        ]
        validate_step_critics(evaluation, group_indices)
        indices.extend(group_indices)
    expected = list(range(1, total_steps + 1))
    if sorted(indices) != expected or len(indices) != len(set(indices)):
        raise PRMValidationError(
            f"final step coverage must contain each index in {expected} exactly once, got {indices}"
        )
