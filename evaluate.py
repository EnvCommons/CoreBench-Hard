"""CORE-Bench scoring — compare agent's report.json against gold answers.

Ported from inspect_evals/core_bench/scorer.py and utils.py. Answers are
compared by type:
  - Numeric: Must fall within a 95% prediction interval (from multiple runs)
  - String: Case-insensitive, trailing punctuation stripped
  - List: Exact match

A task is CORRECT only if ALL questions are answered correctly.
"""

import math
import string
from typing import Any

import numpy as np
from scipy.stats import t as t_dist


def categorize_keys(
    gt_result: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    """Group keys by type from the first ground-truth record."""
    single_gt = gt_result[0]
    numeric_keys = [k for k, v in single_gt.items() if isinstance(v, (int, float))]
    list_keys = [k for k, v in single_gt.items() if isinstance(v, list)]
    string_keys = [k for k, v in single_gt.items() if isinstance(v, str)]
    return numeric_keys, list_keys, string_keys


def count_questions(
    numeric_keys: list[str], list_keys: list[str], string_keys: list[str]
) -> tuple[int, int]:
    """Count total written and vision questions."""
    all_keys = numeric_keys + list_keys + string_keys
    total_written = sum(1 for k in all_keys if "fig" not in k)
    total_vision = sum(1 for k in all_keys if "fig" in k)
    return total_written, total_vision


def clean_agent_results(agent_result: dict) -> dict[str, Any]:
    """Convert agent result values to float where possible."""
    if not isinstance(agent_result, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in agent_result.items():
        try:
            if isinstance(value, str) and "%" in value:
                value = value.replace("%", "")
            try:
                cleaned[key] = float(value)
            except (ValueError, TypeError):
                cleaned[key] = value
        except Exception:
            cleaned[key] = value
    return cleaned


def strip_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Strip trailing punctuation from dictionary keys."""
    return {k.rstrip(string.punctuation): v for k, v in d.items()}


def calculate_prediction_intervals(
    gt_result: list[dict[str, Any]], numeric_keys: list[str]
) -> dict[str, tuple[float, float]]:
    """Compute 95% prediction intervals for each numeric key."""
    intervals = {}
    num_trials = len(gt_result)
    t_value = t_dist.ppf(0.975, max(num_trials - 1, 1))
    for key in numeric_keys:
        values = [trial[key] for trial in gt_result]
        mean_val = np.mean(values)
        std_dev = np.std(values, ddof=1) if num_trials > 1 else 0.0
        margin = t_value * std_dev * math.sqrt(1 + 1 / num_trials)
        intervals[key] = (mean_val - margin, mean_val + margin)
    return intervals


def round_to_gt_precision(gt: float, agent_value: float) -> float:
    """Round agent_value to the same decimal precision as gt."""
    gt_str = str(gt)
    if "." in gt_str:
        decimal_places = len(gt_str.split(".")[1])
    else:
        decimal_places = 0
    return round(agent_value, decimal_places)


def check_numeric_answer(
    agent_value: Any, interval: tuple[float, float]
) -> bool:
    """Check if agent's numeric answer falls within the prediction interval."""
    if isinstance(agent_value, str):
        try:
            agent_value = float(agent_value)
        except ValueError:
            return False
    if not isinstance(agent_value, (int, float)):
        return False
    lower, upper = interval
    return (
        lower <= round_to_gt_precision(lower, agent_value)
        and round_to_gt_precision(upper, agent_value) <= upper
    )


def evaluate_results(
    agent_result: dict[str, Any], gt_result: list[dict[str, Any]]
) -> dict[str, int]:
    """Compare agent results against gold answers.

    Returns dict with correct/total counts for written and vision questions.
    """
    gt_result = [strip_keys(record) for record in gt_result]
    numeric_keys, list_keys, string_keys = categorize_keys(gt_result)
    total_written, total_vision = count_questions(numeric_keys, list_keys, string_keys)
    clean_results = strip_keys(clean_agent_results(agent_result))
    pred_intervals = calculate_prediction_intervals(gt_result, numeric_keys)

    correct_written = 0
    correct_vision = 0
    gt_task_questions = gt_result[0].keys()

    for agent_key, agent_val in clean_results.items():
        if agent_key not in gt_task_questions:
            continue

        correct = False
        if agent_key in numeric_keys and agent_key in pred_intervals:
            correct = check_numeric_answer(agent_val, pred_intervals[agent_key])
        elif agent_key in list_keys:
            correct = agent_val == gt_result[0][agent_key]
        elif agent_key in string_keys:
            correct = (
                str(agent_val).lower().rstrip(string.punctuation)
                == str(gt_result[0][agent_key]).lower().rstrip(string.punctuation)
            )

        if correct:
            if "fig" in agent_key:
                correct_vision += 1
            else:
                correct_written += 1

    return {
        "correct_written_answers": correct_written,
        "correct_vision_answers": correct_vision,
        "total_written_questions": total_written,
        "total_vision_questions": total_vision,
    }
