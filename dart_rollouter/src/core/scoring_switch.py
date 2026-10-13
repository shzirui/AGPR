"""Select the process-reward scorer. Default is v2.1; v2.2 is opt-in."""

from types import SimpleNamespace


def resolve_process_reward_scoring(version) -> SimpleNamespace:
    name = str(version or "v2_1").strip().lower().replace(".", "_").replace("-", "_")
    if name in {"v2_1", "v21", "2_1"}:
        from src.core.group import async_group_steps
        from src.core.group_multi import async_group_steps as async_group_steps_multi
        from src.core.prm_validation import (
            PRMValidationError,
            derive_group_score,
            validate_final_step_coverage,
            validate_grouped_step_coverage,
        )
        from src.core.score_backward_multi import (
            async_refine_group_judgments_multi_turn,
            async_score_recursive_step_multi_turn,
        )
        from src.core.score_backward_single import (
            async_refine_group_judgments_single_turn,
            async_score_recursive_step_single_turn,
        )

        def group_score(conclusion):
            return derive_group_score(
                conclusion["subgoal_status"],
                conclusion["goal_contribution"],
            )

        def refined_score(item):
            return derive_group_score(
                item["refined_subgoal_status"],
                item["refined_goal_contribution"],
            )

        def previous_group_line(index, subgoal, conclusion, score):
            summary = conclusion.get("history_summary_for_next_group") or conclusion.get("reason", "")
            return (
                f"Group {index + 1}:\n"
                f"  subgoal: {subgoal.get('subgoal_name', 'Unknown')}\n"
                f"  reasoning: {subgoal.get('reasoning', 'Unknown')}\n"
                f"  score: {score}\n"
                f"  summary: {summary}\n"
                f"  status: {conclusion.get('subgoal_status', 'Unknown')}\n"
                f"  contribution: {conclusion.get('goal_contribution', 'Unknown')}"
            )

        return SimpleNamespace(
            name="v2_1",
            errors=(KeyError, PRMValidationError),
            group_single=async_group_steps,
            group_multi=async_group_steps_multi,
            score_single=async_score_recursive_step_single_turn,
            score_multi=async_score_recursive_step_multi_turn,
            refine_single=async_refine_group_judgments_single_turn,
            refine_multi=async_refine_group_judgments_multi_turn,
            validate_grouped_step_coverage=validate_grouped_step_coverage,
            validate_final_step_coverage=validate_final_step_coverage,
            group_score=group_score,
            refined_score=refined_score,
            previous_group_line=previous_group_line,
        )

    if name in {"v2_2", "v22", "2_2"}:
        from src.core.group_multi import async_group_steps as async_group_steps_multi
        from src.core.scoring_v2_2.group import async_group_steps
        from src.core.scoring_v2_2.prm_validation import (
            PRMValidationError,
            validate_final_step_coverage,
            validate_grouped_step_coverage,
        )
        from src.core.scoring_v2_2.score_backward_multi import (
            async_refine_group_scores_multi_turn,
            async_score_recursive_step_multi_turn,
        )
        from src.core.scoring_v2_2.score_backward_single import (
            async_refine_group_scores_single_turn,
            async_score_recursive_step_single_turn,
        )

        def group_score(conclusion):
            try:
                score = float(conclusion.get("score"))
            except (TypeError, ValueError) as error:
                raise PRMValidationError("group score must be numeric") from error
            return score

        def refined_score(item):
            try:
                score = float(item.get("refined_score"))
            except (TypeError, ValueError) as error:
                raise PRMValidationError("refined score must be numeric") from error
            return score

        def previous_group_line(index, subgoal, conclusion, score):
            summary = conclusion.get("history_summary_for_next_group") or conclusion.get("reason", "")
            contribution = conclusion.get("goal_contribution", conclusion.get("global_alignment", "Unknown"))
            return (
                f"G{index + 1} {subgoal.get('subgoal_name', 'Unknown')}: "
                f"{conclusion.get('subgoal_status', 'Unknown')} + {contribution}, "
                f"score={score}. {summary}"
            )

        return SimpleNamespace(
            name="v2_2",
            errors=(KeyError, PRMValidationError),
            group_single=async_group_steps,
            group_multi=async_group_steps_multi,
            score_single=async_score_recursive_step_single_turn,
            score_multi=async_score_recursive_step_multi_turn,
            refine_single=async_refine_group_scores_single_turn,
            refine_multi=async_refine_group_scores_multi_turn,
            validate_grouped_step_coverage=validate_grouped_step_coverage,
            validate_final_step_coverage=validate_final_step_coverage,
            group_score=group_score,
            refined_score=refined_score,
            previous_group_line=previous_group_line,
        )

    raise ValueError(
        f"Unsupported process_reward.scoring_version={version!r}; expected v2_1 or v2_2"
    )
