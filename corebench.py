"""CORE-Bench Hard — OpenReward sandbox environment for computational reproducibility.

Agents must reproduce results from scientific code repositories. For the "hard"
difficulty, agents get source code but NOT reproduction instructions or
pre-computed results — they must figure out how to install dependencies, run the
code, and answer questions about the output.

Paper: https://arxiv.org/abs/2409.11353
Dataset: https://huggingface.co/datasets/siegelz/core-bench
"""

import json
import logging
import os
from pathlib import Path

from openreward import AsyncOpenReward, SandboxBucketConfig, SandboxSettings
from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool
from pydantic import BaseModel

from evaluate import evaluate_results

logger = logging.getLogger(__name__)

# --- Module-level data loading ---

if os.path.exists("/orwd_data"):
    _DATA_DIR = Path("/orwd_data")
else:
    _DATA_DIR = Path(__file__).parent

_all_records: dict[str, dict] = {}
_train_tasks: list[JSONObject] = []
_test_tasks: list[JSONObject] = []

for _split_name, _task_list in [("train", _train_tasks), ("test", _test_tasks)]:
    _json_path = _DATA_DIR / f"tasks_{_split_name}.json"
    if not _json_path.exists():
        logger.warning(f"Data file not found: {_json_path}")
        continue
    with open(_json_path) as _f:
        _records = json.load(_f)
    for _record in _records:
        _record_id = _record["id"]
        _all_records[_record_id] = _record
        # Task spec: agent-visible fields only (no gold answers)
        _task_list.append({
            "id": _record_id,
            "field": _record["field"],
            "language": _record["language"],
        })


# --- Pydantic parameter models ---

class BashParams(BaseModel, extra="forbid"):
    command: str


class SubmitParams(BaseModel, extra="forbid"):
    """Submit answers by providing the path to report.json (or leave default)."""
    pass


# --- Environment class ---

class CoreBenchHard(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)

        record_id = str(task_spec["id"])
        if record_id not in _all_records:
            raise ValueError(f"Unknown task id: {record_id}")

        record = _all_records[record_id]
        self.capsule_id: str = record["id"]
        self.task_prompt: str = record["task_prompt"]
        self.gold_results: list[dict] = record["results"]
        self.field: str = record["field"]
        self.language: str = record["language"]

        # Question fields the agent must answer
        self.json_fields = list(self.gold_results[0].keys())

        api_key = (
            secrets.get("OPENREWARD_API_KEY")
            or secrets.get("api_key")
            or os.environ.get("OPENREWARD_API_KEY", "").strip('"')
        )
        if not api_key:
            raise ValueError("OpenReward API key required (pass as OPENREWARD_API_KEY)")

        self.sandbox_settings = SandboxSettings(
            environment="siegelz/CoreBench-Hard",
            image="generalreasoning/python-ds:3.12-tools",
            machine_size="1:4",
            block_network=False,
            bucket_config=SandboxBucketConfig(
                mount_path="/home/ubuntu/capsule_data",
                read_only=True,
                only_dir=f"sandbox_data/{self.capsule_id}/",
            ),
        )

        or_client = AsyncOpenReward(api_key=api_key)
        self.sandbox = or_client.sandbox(self.sandbox_settings)

        self.submitted = False

    async def setup(self) -> None:
        await self.sandbox.start()

        # Extract capsule tarball from bucket mount, then clean up for "hard" difficulty
        # Tarball is mounted at /home/ubuntu/capsule_data/{capsule_id}.tar.gz
        setup_cmd = (
            f"cd /home/ubuntu && "
            f"tar --no-same-owner --strip-components=1 -xzf /home/ubuntu/capsule_data/{self.capsule_id}.tar.gz && "
            # Hard difficulty: remove reproduction docs, environment dir, run scripts
            # Keep source code and README so agent can figure it out
            f"rm -f REPRODUCING.md run run.sh && rm -rf environment results"
        )
        result = await self.sandbox.run(setup_cmd)
        output, code = result
        if code != 0:
            logger.error(f"Setup failed for {self.capsule_id}: {output}")
            raise RuntimeError(f"Failed to set up capsule {self.capsule_id}: {output}")

    async def teardown(self) -> None:
        await self.sandbox.stop()

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            return _train_tasks
        elif split == "test":
            return _test_tasks
        return []

    async def get_prompt(self) -> list[TextBlock]:
        prompt = f"""Task: codeocean_hard

Your goal is to test the computational reproducibility of the repository in your
current directory (/home/ubuntu), which is code from a scientific paper.

Specifically, you need to {self.task_prompt}

Save your report to a file named **report.json** in /home/ubuntu/ where you fill
in all of the following fields: {self.json_fields}

You should install all of the requirements found in the Readme file and then run
the commands necessary to answer the questions.

## Environment

- Field: {self.field}
- Language: {self.language}
- Working directory: /home/ubuntu (capsule code is extracted here)
- Network access: enabled (you can install packages)

Use the `bash` tool to explore the code, install dependencies, run scripts, and
create report.json. When done, use the `submit` tool to evaluate your answers."""

        return [TextBlock(text=prompt)]

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """Execute a bash command in the sandbox environment."""
        result = await self.sandbox.run(params.command.strip())
        output, code = result

        if result.truncated:
            output = f"...(truncated, output exceeded limit)\n{output}"

        return ToolOutput(
            blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
            metadata={"output": output, "exit_code": code, "truncated": result.truncated},
            reward=0.0,
            finished=False,
        )

    @tool
    async def submit(self, params: SubmitParams) -> ToolOutput:
        """Submit your report.json for evaluation.

        Reads /home/ubuntu/report.json and compares your answers against the
        gold-standard results. This is a terminal action — one attempt only.
        """
        if self.submitted:
            return ToolOutput(
                blocks=[TextBlock(text="Already submitted. Only one submission is allowed.")],
                metadata={"error": "already_submitted"},
                reward=0.0,
                finished=True,
            )

        self.submitted = True

        # Read report.json from sandbox
        try:
            result = await self.sandbox.run("cat /home/ubuntu/report.json")
            report_content, code = result
            if code != 0:
                return ToolOutput(
                    blocks=[TextBlock(text="Error: report.json not found at /home/ubuntu/report.json")],
                    metadata={"error": "report_not_found"},
                    reward=0.0,
                    finished=True,
                )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(text=f"Error reading report.json: {e}")],
                metadata={"error": str(e)},
                reward=0.0,
                finished=True,
            )

        # Parse JSON
        try:
            agent_result = json.loads(report_content)
        except json.JSONDecodeError as e:
            return ToolOutput(
                blocks=[TextBlock(text=f"Error: report.json is not valid JSON: {e}")],
                metadata={"error": "invalid_json"},
                reward=0.0,
                finished=True,
            )

        # Evaluate
        try:
            score_detail = evaluate_results(agent_result, self.gold_results)
        except Exception as e:
            logger.exception("Evaluation error")
            return ToolOutput(
                blocks=[TextBlock(text=f"Evaluation error: {e}")],
                metadata={"error": str(e)},
                reward=0.0,
                finished=True,
            )

        total = score_detail["total_written_questions"] + score_detail["total_vision_questions"]
        correct = score_detail["correct_written_answers"] + score_detail["correct_vision_answers"]
        all_correct = (
            score_detail["correct_written_answers"] == score_detail["total_written_questions"]
            and score_detail["correct_vision_answers"] == score_detail["total_vision_questions"]
        )

        # Binary reward: all-or-nothing (matching original CORE-Bench methodology)
        reward = 1.0 if all_correct else 0.0

        result_text = f"""Submission Results:
- Correct: {correct}/{total} questions
- Written: {score_detail['correct_written_answers']}/{score_detail['total_written_questions']}
- Vision: {score_detail['correct_vision_answers']}/{score_detail['total_vision_questions']}
- All correct: {all_correct}
- Reward: {reward}"""

        return ToolOutput(
            blocks=[TextBlock(text=result_text)],
            metadata={
                "reward": reward,
                "all_correct": all_correct,
                **score_detail,
            },
            reward=reward,
            finished=True,
        )


if __name__ == "__main__":
    from openreward.environments import Server
    Server([CoreBenchHard]).run()
