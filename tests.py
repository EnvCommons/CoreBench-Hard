"""Unit tests for the CORE-Bench Hard OpenReward environment."""

import json
from pathlib import Path

import pytest

from evaluate import (
    categorize_keys,
    check_numeric_answer,
    clean_agent_results,
    evaluate_results,
    calculate_prediction_intervals,
    strip_keys,
)


# --- Scoring helper tests ---


class TestCategorizeKeys:
    def test_basic(self):
        gt = [{"score": 0.95, "name": "test", "items": [1, 2]}]
        numeric, lists, strings = categorize_keys(gt)
        assert numeric == ["score"]
        assert lists == ["items"]
        assert strings == ["name"]

    def test_empty(self):
        gt = [{}]
        numeric, lists, strings = categorize_keys(gt)
        assert numeric == [] and lists == [] and strings == []


class TestCleanAgentResults:
    def test_numeric_string(self):
        result = clean_agent_results({"score": "0.95"})
        assert result["score"] == 0.95

    def test_percentage(self):
        result = clean_agent_results({"rate": "85%"})
        assert result["rate"] == 85.0

    def test_non_numeric(self):
        result = clean_agent_results({"name": "hello"})
        assert result["name"] == "hello"

    def test_non_dict(self):
        assert clean_agent_results("not a dict") == {}


class TestStripKeys:
    def test_strips_punctuation(self):
        result = strip_keys({"question?": "answer", "clean": "val"})
        assert "question" in result
        assert "clean" in result


class TestCheckNumericAnswer:
    def test_within_interval(self):
        assert check_numeric_answer(0.95, (0.90, 1.00)) is True

    def test_outside_interval(self):
        assert check_numeric_answer(0.5, (0.90, 1.00)) is False

    def test_string_numeric(self):
        assert check_numeric_answer("0.95", (0.90, 1.00)) is True

    def test_invalid_string(self):
        assert check_numeric_answer("not a number", (0.0, 1.0)) is False


class TestPredictionIntervals:
    def test_single_run(self):
        gt = [{"score": 0.95}]
        intervals = calculate_prediction_intervals(gt, ["score"])
        # Single value → interval is just that value (std=0, margin=0)
        lower, upper = intervals["score"]
        assert lower == upper == 0.95

    def test_multiple_runs(self):
        gt = [{"score": 0.90}, {"score": 0.95}, {"score": 1.00}]
        intervals = calculate_prediction_intervals(gt, ["score"])
        lower, upper = intervals["score"]
        assert lower < 0.90  # Interval extends below min
        assert upper > 1.00  # Interval extends above max


class TestEvaluateResults:
    def test_all_correct_numeric(self):
        gt = [{"score": 0.95}]
        agent = {"score": 0.95}
        result = evaluate_results(agent, gt)
        assert result["correct_written_answers"] == 1
        assert result["total_written_questions"] == 1

    def test_wrong_numeric(self):
        gt = [{"score": 0.95}]
        agent = {"score": 0.5}
        result = evaluate_results(agent, gt)
        assert result["correct_written_answers"] == 0

    def test_string_case_insensitive(self):
        gt = [{"name": "Hello World"}]
        agent = {"name": "hello world"}
        result = evaluate_results(agent, gt)
        assert result["correct_written_answers"] == 1

    def test_missing_key(self):
        gt = [{"score": 0.95, "name": "test"}]
        agent = {"score": 0.95}  # Missing "name"
        result = evaluate_results(agent, gt)
        assert result["correct_written_answers"] == 1  # Only score counted
        assert result["total_written_questions"] == 2

    def test_vision_question(self):
        gt = [{"fig_count": 5}]
        agent = {"fig_count": 5}
        result = evaluate_results(agent, gt)
        assert result["correct_vision_answers"] == 1
        assert result["total_vision_questions"] == 1
        assert result["total_written_questions"] == 0


# --- Task structure tests ---


class TestTaskStructure:
    @pytest.fixture(autouse=True)
    def _load_data(self):
        test_path = Path(__file__).parent / "tasks_test.json"
        train_path = Path(__file__).parent / "tasks_train.json"
        if not test_path.exists() or not train_path.exists():
            pytest.skip("Data files not found — run prepare_data.py first")
        with open(test_path) as f:
            self.test_tasks = json.load(f)
        with open(train_path) as f:
            self.train_tasks = json.load(f)

    def test_test_count(self):
        assert len(self.test_tasks) == 45

    def test_train_count(self):
        assert len(self.train_tasks) == 45

    def test_required_fields(self):
        required = {"id", "field", "language", "task_prompt", "results"}
        for task in self.test_tasks + self.train_tasks:
            assert required.issubset(set(task.keys())), f"Missing fields in {task['id']}"

    def test_all_have_results(self):
        for task in self.test_tasks + self.train_tasks:
            assert len(task["results"]) > 0, f"No results for {task['id']}"
            assert isinstance(task["results"][0], dict)

    def test_stable_ordering(self):
        test_ids = [t["id"] for t in self.test_tasks]
        train_ids = [t["id"] for t in self.train_tasks]
        assert test_ids == sorted(test_ids)
        assert train_ids == sorted(train_ids)

    def test_no_id_overlap(self):
        test_ids = set(t["id"] for t in self.test_tasks)
        train_ids = set(t["id"] for t in self.train_tasks)
        assert len(test_ids & train_ids) == 0


# --- Environment class tests ---


class TestCoreBenchEnv:
    @pytest.fixture(autouse=True)
    def _check_data(self):
        if not (Path(__file__).parent / "tasks_test.json").exists():
            pytest.skip("Data files not found — run prepare_data.py first")

    def test_list_splits(self):
        from corebench import CoreBenchHard
        assert CoreBenchHard.list_splits() == ["train", "test"]

    def test_list_tasks_test(self):
        from corebench import CoreBenchHard
        tasks = CoreBenchHard.list_tasks("test")
        assert len(tasks) == 45

    def test_list_tasks_train(self):
        from corebench import CoreBenchHard
        tasks = CoreBenchHard.list_tasks("train")
        assert len(tasks) == 45

    def test_task_spec_fields(self):
        from corebench import CoreBenchHard
        tasks = CoreBenchHard.list_tasks("test")
        for task in tasks[:5]:
            assert "id" in task
            assert "field" in task
            assert "language" in task
            # Should NOT expose gold answers
            assert "results" not in task
            assert "task_prompt" not in task
