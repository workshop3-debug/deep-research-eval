"""
DeepAgent Evaluation with Inspect AI
======================================
Evaluates the full DeepAgent pipeline as well as each subagent individually:
  - research()   - read-only data gathering from the SCADS2025 dataset path
  - plan()       - structured planning / analysis
  - general()    - full-capability execution

Samples are loaded from: deepagent_samples.json (same directory as this file)
Each sample must have: id, role, input, target, and a top-level 'role' key.

Run examples
------------
# Full suite (all four tasks):
    inspect eval deepagent_eval.py --model anthropic/claude-sonnet-4-20250514

# A single task:
    inspect eval deepagent_eval.py@task_research_agent --model anthropic/claude-sonnet-4-20250514
    inspect eval deepagent_eval.py@task_plan_agent     --model anthropic/claude-sonnet-4-20250514
    inspect eval deepagent_eval.py@task_general_agent  --model anthropic/claude-sonnet-4-20250514
    inspect eval deepagent_eval.py@task_deepagent_full --model anthropic/claude-sonnet-4-20250514

# View results afterwards:
    inspect view
"""

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

from inspect_ai import Task, task
from inspect_ai.agent import as_solver, deepagent, general, plan, research
from inspect_ai.dataset import Sample
from inspect_ai.scorer import (
    Score,
    Target,
    accuracy,
    model_graded_fact,
    model_graded_qa,
    scorer,
)
from inspect_ai.solver import TaskState
from inspect_ai.tool import bash, grep, list_files, read_file, text_editor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLES_FILE = Path(__file__).parent / "deepagent_samples.json"

DATASET_PATH = (
    "/home/workshop11/efs/resources/datasets/SCADS2025/ZendiaDatasets/Clean Datasets"
)

JUDGE_MODEL = "openai/openai/gpt-4o-mini"

# ---------------------------------------------------------------------------
# Load samples from JSON
# ---------------------------------------------------------------------------

def load_samples(role: str) -> list[Sample]:
    """
    Read deepagent_samples.json and return samples matching `role`.
    Each JSON object must have: id, input, target, and a top-level 'role' key.
    """
    with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)

    return [
        Sample(
            id=entry["id"],
            input=entry["input"],
            target=entry["target"],
            metadata=entry.get("metadata", {}),
        )
        for entry in raw
        if entry.get("role") == role
    ]


# ---------------------------------------------------------------------------
# Custom scorer: task-completion rubric via LLM judge
# ---------------------------------------------------------------------------

@scorer(metrics=[accuracy()])
def task_completion_scorer(model: str = JUDGE_MODEL):
    """
    LLM-as-judge that rates the agent response on a 0 / 0.5 / 1.0 scale.
      C (correct / complete)  -> 1.0
      P (partial)             -> 0.5
      I (incorrect / missing) -> 0.0
    """
    async def score(state: TaskState, target: Target) -> Score:
        judge = model_graded_qa(
            model=model,
            instructions=dedent("""\
                You are an expert evaluator.

                Rate the ASSISTANT RESPONSE against the CRITERION below.

                Scoring rubric:
                  C (correct / complete)   - fully satisfies the criterion
                  P (partial)              - partially satisfies, some gaps
                  I (incorrect/missing)    - does not satisfy the criterion

                Respond with exactly one letter: C, P, or I
            """),
        )
        result = await judge(state, target)
        grade_map = {"C": 1.0, "P": 0.5, "I": 0.0}
        value = grade_map.get(str(result.value).strip().upper(), 0.0)
        return Score(
            value=value,
            explanation=result.explanation,
            metadata={"raw_grade": result.value},
        )

    return score


# ---------------------------------------------------------------------------
# Tool sets
# ---------------------------------------------------------------------------

READONLY_TOOLS  = [read_file(), list_files(), grep()]
READWRITE_TOOLS = [read_file(), list_files(), grep(), bash(), text_editor()]


# ---------------------------------------------------------------------------
# Task 1 - Research subagent (standalone)
# ---------------------------------------------------------------------------

@task
def task_research_agent() -> Task:
    """Evaluate the research() subagent in isolation."""
    research_subagent = research(
        tools=READONLY_TOOLS,
        instructions=dedent(f"""\
            You have read-only access to the dataset directory:
            {DATASET_PATH}

            Use list_files, read_file, and grep to gather information.
            Never attempt to write or modify files.
        """),
    )

    # Correct pattern: instantiate deepagent(...) first, then wrap with as_solver()
    agent = deepagent(
        subagents=[research_subagent],
        memory=False,
        instructions=(
            f"Delegate ALL work to the research subagent. Dataset: {DATASET_PATH}"
        ),
    )

    return Task(
        dataset=load_samples("research"),
        solver=as_solver(agent),
        scorer=[
            task_completion_scorer(),
            model_graded_fact(model=JUDGE_MODEL),
        ],
        metadata={"subagent": "research"},
        sandbox="local",
    )


# ---------------------------------------------------------------------------
# Task 2 - Plan subagent (standalone)
# ---------------------------------------------------------------------------

@task
def task_plan_agent() -> Task:
    """Evaluate the plan() subagent in isolation."""
    plan_subagent = plan(
        tools=READONLY_TOOLS,
        instructions=dedent(f"""\
            You are a planning specialist. The dataset you may inspect
            (read-only) lives at: {DATASET_PATH}

            Produce structured, actionable plans. Do not execute code or
            modify any files.
        """),
    )

    agent = deepagent(
        subagents=[plan_subagent],
        memory=False,
        instructions=(
            f"Delegate ALL planning work to the plan subagent. Dataset: {DATASET_PATH}"
        ),
    )

    return Task(
        dataset=load_samples("plan"),
        solver=as_solver(agent),
        scorer=[
            task_completion_scorer(),
            model_graded_fact(model=JUDGE_MODEL),
        ],
        metadata={"subagent": "plan"},
        sandbox="local",
    )


# ---------------------------------------------------------------------------
# Task 3 - General subagent (standalone)
# ---------------------------------------------------------------------------

@task
def task_general_agent() -> Task:
    """Evaluate the general() subagent in isolation."""
    general_subagent = general(
        tools=READWRITE_TOOLS,
        instructions=dedent(f"""\
            You are a capable general-purpose assistant.
            Dataset directory: {DATASET_PATH}

            You may read AND process files (bash, text_editor). Use Python
            via bash for data analysis when appropriate.
        """),
        memory="readwrite",
    )

    agent = deepagent(
        subagents=[general_subagent],
        memory=True,
        instructions=(
            f"Delegate ALL execution work to the general subagent. Dataset: {DATASET_PATH}"
        ),
    )

    return Task(
        dataset=load_samples("general"),
        solver=as_solver(agent),
        scorer=[
            task_completion_scorer(),
            model_graded_fact(model=JUDGE_MODEL),
        ],
        metadata={"subagent": "general"},
        sandbox="local",        
    )


# ---------------------------------------------------------------------------
# Task 4 - Full DeepAgent (all three subagents collaborating)
# ---------------------------------------------------------------------------

@task
def task_deepagent_full() -> Task:
    """
    Evaluate the complete DeepAgent with all three subagents:
    research -> plan -> general.
    """
    research_subagent = research(
        tools=READONLY_TOOLS,
        instructions=dedent(f"""\
            Your role: read-only data exploration.
            Dataset path: {DATASET_PATH}

            Use list_files, read_file, and grep.
            Summarise findings clearly for the planning stage.
        """),
    )

    plan_subagent = plan(
        tools=READONLY_TOOLS,
        instructions=dedent(f"""\
            Your role: structured, actionable planning based on research findings.
            Dataset path (reference only): {DATASET_PATH}

            Output numbered steps with clear success criteria.
        """),
    )

    general_subagent = general(
        tools=READWRITE_TOOLS,
        instructions=dedent(f"""\
            Your role: execute the plan from the plan subagent.
            Dataset path: {DATASET_PATH}

            Use bash for computations. Return results as a structured
            markdown report.
        """),
        memory="readwrite",
    )

    agent = deepagent(
        subagents=[research_subagent, plan_subagent, general_subagent],
        memory=True,
        todo_write=True,
        max_depth=2,
        instructions=dedent(f"""\
            You orchestrate three specialised subagents:
              - research  : read-only data exploration
              - plan      : structured planning
              - general   : code execution and file operations

            Dataset path: {DATASET_PATH}

            Always follow the sequence: research -> plan -> general.
            Consolidate outputs into a final markdown report.
        """),
    )

    return Task(
        dataset=load_samples("deepagent"),
        solver=as_solver(agent),
        scorer=[
            task_completion_scorer(),
            model_graded_qa(
                model=JUDGE_MODEL,
                instructions=dedent("""\
                    Does the final report:
                      (a) cover data exploration findings?
                      (b) include a clear analysis plan?
                      (c) contain computed statistics or test results?
                    Answer C (all three present), P (one or two), or I (none).
                """),
            ),
        ],
        metadata={"subagent": "deepagent_full"},
        sandbox="local",        
    )


# ---------------------------------------------------------------------------
# To run ALL tasks in one sweep, just point inspect at this file with no
# task name - it will auto-discover all @task functions:
#
#   inspect eval deepagent_eval.py --model anthropic/claude-sonnet-4-20250514
#
# Note: returning list[Task] from a @task function is not supported by the
# inspect registry, so we don't define a combined task here.
# ---------------------------------------------------------------------------