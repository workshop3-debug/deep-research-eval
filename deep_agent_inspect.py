"""
DeepAgent Evaluation with Inspect AI
======================================
Evaluates the full DeepAgent pipeline as well as each subagent individually
against the Zendia intelligence corpus:
  - research()   - read-only data gathering from the SCADS2025 dataset path
  - plan()       - structured planning / analysis
  - general()    - full-capability execution

Samples are loaded from: deepagent_samples.json (same directory as this file)
Each sample must have: id, role, input, target, metadata.

Run examples
------------
# Full suite (all four tasks):
    inspect eval deep_agent_inspect.py --model openai/gpt-5-mini

# A single task:
    inspect eval deep_agent_inspect.py@task_research_agent --model openai/gpt-5-mini
    inspect eval deep_agent_inspect.py@task_plan_agent     --model openai/gpt-5-mini
    inspect eval deep_agent_inspect.py@task_general_agent  --model openai/gpt-5-mini
    inspect eval deep_agent_inspect.py@task_deepagent_full --model openai/gpt-5-mini

# View results afterwards:
    inspect view
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from textwrap import dedent
from typing import Any

from inspect_ai import Task, task
from inspect_ai.agent import as_solver, deepagent, general, plan, research
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    mean,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from inspect_ai.tool import bash, grep, list_files, read_file, text_editor

# ---------------------------------------------------------------------------
# LLM proxy wiring — LAS proxy speaks the OpenAI protocol.
# ---------------------------------------------------------------------------

if os.getenv("OPENAI_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]
os.environ.setdefault("OPENAI_BASE_URL", "https://llm-west.ncsu-las.net/v1")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLES_FILE = Path(__file__).parent / "deepagent_samples.json"

DATASET_PATH = os.getenv(
    "DEEP_RESEARCH_DATASET",
    "/home/workshop3/efs/resources/datasets/SCADS2025/ZendiaDatasets/Clean Datasets",
)

JUDGE_MODEL = os.getenv("DEEP_RESEARCH_JUDGE_MODEL", "openai/gpt-5-mini")


# ---------------------------------------------------------------------------
# Load samples from JSON
# ---------------------------------------------------------------------------

def load_samples(role: str) -> list[Sample]:
    """Read deepagent_samples.json and return samples matching `role`."""
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
# LLM-judge helpers (shared rubric infrastructure)
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def _parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


JUDGE_TEMPLATE = """You are a strict evaluator. Score the assistant response on \
a 1-5 scale and respond with valid JSON only:
{{"score": <1-5 integer>, "rationale": "<1-3 sentences>"}}

Criteria:
{criteria}

Original task:
{task}

Reference target (what a good response should cover):
{target}

Assistant response:
{response}
"""


async def _judge(criteria: str, task_text: str, target_text: str, response: str) -> tuple[float, str]:
    out = await get_model(JUDGE_MODEL).generate(
        [
            ChatMessageSystem(
                content="You are a strict evaluator. Respond with valid JSON only."
            ),
            ChatMessageUser(
                content=JUDGE_TEMPLATE.format(
                    criteria=criteria,
                    task=task_text,
                    target=target_text,
                    response=response or "(empty response)",
                )
            ),
        ],
        config=GenerateConfig(temperature=0.0),
    )
    data = _parse_json(out.completion) or {}
    try:
        raw = float(data.get("score", 0))
    except (TypeError, ValueError):
        raw = 0.0
    score = max(0.0, min(5.0, raw)) / 5.0  # normalize to [0, 1]
    return score, str(data.get("rationale", "judge returned no rationale"))


# Per-role rubrics — analogous to the phase scorers in deep_research_inspect.py.
RESEARCH_CRITERIA = """- Faithfulness: every claim is supported by evidence drawn from the corpus.
- Relevance: the answer addresses what was actually asked.
- Coverage: uses multiple relevant reports rather than a single hit.
- Citations: cites specific report files / serials so claims are traceable.
- Honest gaps: explicitly says when the corpus is silent on something.
- Discipline: stays read-only and does not fabricate file contents."""

PLAN_CRITERIA = """- Structure: clearly organized phases or numbered steps.
- Specificity: steps are actionable, not vague aspirations.
- Coverage: addresses every dimension the prompt asks for (deliverables, dependencies, risks, etc.).
- Atomicity: sub-questions or steps are focused enough to execute independently.
- Success criteria: defines how each step is judged 'done'.
- No execution: produces a plan only, without running code or writing files."""

GENERAL_CRITERIA = """- Correctness: computed values and extracted facts match what the corpus actually contains.
- Completeness: every sub-task in the prompt is attempted.
- Faithfulness: claims are grounded in observed files, not invented.
- Format compliance: output matches the requested format (markdown table, word limits, ordering).
- Citations: when asked, references specific report serials / paths.
- Synthesis: where the prompt asks for interpretation, the synthesis is supported by the extracted data."""

DEEPAGENT_CRITERIA = """- Pipeline: the response shows evidence of research -> plan -> execution stages.
- Faithfulness: claims trace back to specific reports in the corpus.
- Structure: final output is a coherent markdown brief with executive summary, thematic sections, and sources.
- Coverage: integrates findings across multiple sub-questions / reports.
- Citations: report serials or paths are cited inline or in a sources list.
- Analysis quality: where hypotheses are required, each is rated against evidence."""


def _rubric_scorer(criteria: str, label: str) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion or ""
        task_text = state.input_text
        target_text = target.text if hasattr(target, "text") else str(target)
        s, rationale = await _judge(criteria, task_text, target_text, response)
        return Score(value=s, explanation=rationale, metadata={"rubric": label})

    return score


@scorer(metrics=[mean(), stderr()])
def score_research_rubric() -> Scorer:
    return _rubric_scorer(RESEARCH_CRITERIA, "research")


@scorer(metrics=[mean(), stderr()])
def score_plan_rubric() -> Scorer:
    return _rubric_scorer(PLAN_CRITERIA, "plan")


@scorer(metrics=[mean(), stderr()])
def score_general_rubric() -> Scorer:
    return _rubric_scorer(GENERAL_CRITERIA, "general")


@scorer(metrics=[mean(), stderr()])
def score_deepagent_rubric() -> Scorer:
    return _rubric_scorer(DEEPAGENT_CRITERIA, "deepagent")


# Lightweight C/P/I task-completion scorer kept for backwards compatibility.
CPI_INSTRUCTIONS = """Decide whether the assistant response satisfies the task \
and its target description.
Reply with exactly one letter:
  C - fully satisfies the target
  P - partially satisfies, with material gaps
  I - does not satisfy the target"""


@scorer(metrics=[mean(), stderr()])
def task_completion_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion or ""
        task_text = state.input_text
        target_text = target.text if hasattr(target, "text") else str(target)
        out = await get_model(JUDGE_MODEL).generate(
            [
                ChatMessageSystem(content="You are a strict evaluator."),
                ChatMessageUser(
                    content=dedent(
                        f"""\
                        {CPI_INSTRUCTIONS}

                        Task:
                        {task_text}

                        Target:
                        {target_text}

                        Response:
                        {response or '(empty response)'}

                        Reply with exactly one letter: C, P, or I.
                        """
                    )
                ),
            ],
            config=GenerateConfig(temperature=0.0),
        )
        letter = (out.completion or "").strip().upper()[:1]
        value = {"C": 1.0, "P": 0.5, "I": 0.0}.get(letter, 0.0)
        return Score(
            value=value,
            explanation=out.completion.strip(),
            metadata={"raw_grade": letter},
        )

    return score


# ---------------------------------------------------------------------------
# Tool sets
# ---------------------------------------------------------------------------

READONLY_TOOLS = [read_file(), list_files(), grep()]
READWRITE_TOOLS = [read_file(), list_files(), grep(), bash(), text_editor()]


# ---------------------------------------------------------------------------
# Task 1 - Research subagent (standalone)
# ---------------------------------------------------------------------------

@task
def task_research_agent() -> Task:
    """Evaluate the research() subagent on Zendia corpus extraction tasks."""
    research_subagent = research(
        tools=READONLY_TOOLS,
        instructions=dedent(f"""\
            You have read-only access to the Zendia intelligence corpus at:
            {DATASET_PATH}

            The corpus is a directory tree of JSON reports. Each report has
            fields like: serial, title, event_date, author, tags, topic,
            classification, body, geo_coordinates.

            Use list_files, read_file, and grep to gather information.
            Cite specific report files (path or serial) when making claims.
            Never attempt to write or modify files.
        """),
    )

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
        scorer=[task_completion_scorer(), score_research_rubric()],
        metadata={"subagent": "research"},
        sandbox="local",
    )


# ---------------------------------------------------------------------------
# Task 2 - Plan subagent (standalone)
# ---------------------------------------------------------------------------

@task
def task_plan_agent() -> Task:
    """Evaluate the plan() subagent on Zendia research-planning tasks."""
    plan_subagent = plan(
        tools=READONLY_TOOLS,
        instructions=dedent(f"""\
            You are a planning specialist for Zendia intelligence analysis.
            The corpus you may inspect (read-only) lives at:
            {DATASET_PATH}

            Produce structured, actionable research plans. Break the task into
            focused sub-questions, list the search terms and report fields you
            would use, and state clear success criteria for each step. Do not
            execute code, modify files, or produce the final analysis itself.
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
        scorer=[task_completion_scorer(), score_plan_rubric()],
        metadata={"subagent": "plan"},
        sandbox="local",
    )


# ---------------------------------------------------------------------------
# Task 3 - General subagent (standalone)
# ---------------------------------------------------------------------------

@task
def task_general_agent() -> Task:
    """Evaluate the general() subagent on Zendia corpus-wide computations."""
    general_subagent = general(
        tools=READWRITE_TOOLS,
        instructions=dedent(f"""\
            You are a capable general-purpose analyst.
            Zendia corpus directory: {DATASET_PATH}

            The corpus is a directory tree of JSON reports with fields like
            serial, title, event_date, tags, topic, classification, body.
            You may read AND process files (bash, text_editor). Use Python
            via bash for aggregation and statistics. Cite specific report
            serials when summarizing findings.
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
        scorer=[task_completion_scorer(), score_general_rubric()],
        metadata={"subagent": "general"},
        sandbox="local",
    )


# ---------------------------------------------------------------------------
# Task 4 - Full DeepAgent (all three subagents collaborating)
# ---------------------------------------------------------------------------

@task
def task_deepagent_full() -> Task:
    """Evaluate the complete DeepAgent: research -> plan -> general."""
    research_subagent = research(
        tools=READONLY_TOOLS,
        instructions=dedent(f"""\
            Your role: read-only exploration of the Zendia intelligence corpus.
            Corpus path: {DATASET_PATH}

            Use list_files, read_file, and grep. Summarise findings clearly
            for the planning stage, citing specific report serials.
        """),
    )

    plan_subagent = plan(
        tools=READONLY_TOOLS,
        instructions=dedent(f"""\
            Your role: structured planning based on the research stage's findings.
            Corpus path (reference only): {DATASET_PATH}

            Output numbered steps with sub-questions to answer, fields to extract,
            and success criteria for each.
        """),
    )

    general_subagent = general(
        tools=READWRITE_TOOLS,
        instructions=dedent(f"""\
            Your role: execute the plan from the plan subagent and produce the
            final report.
            Corpus path: {DATASET_PATH}

            Use bash for aggregations. Produce a markdown intelligence brief
            with an executive summary, thematic sections, and a Sources list
            citing report serials.
        """),
        memory="readwrite",
    )

    agent = deepagent(
        subagents=[research_subagent, plan_subagent, general_subagent],
        memory=True,
        todo_write=True,
        max_depth=2,
        instructions=dedent(f"""\
            You orchestrate three specialised subagents over the Zendia corpus:
              - research  : read-only data exploration
              - plan      : structured planning
              - general   : code execution and report writing

            Corpus path: {DATASET_PATH}

            Always follow the sequence: research -> plan -> general.
            Consolidate outputs into a final markdown intelligence brief with
            executive summary, thematic sections, and a Sources list of report
            serials.
        """),
    )

    return Task(
        dataset=load_samples("deepagent"),
        solver=as_solver(agent),
        scorer=[task_completion_scorer(), score_deepagent_rubric()],
        metadata={"subagent": "deepagent_full"},
        sandbox="local",
    )


# ---------------------------------------------------------------------------
# To run ALL tasks in one sweep, just point inspect at this file with no
# task name - it will auto-discover all @task functions:
#
#   inspect eval deep_agent_inspect.py --model openai/gpt-5-mini
# ---------------------------------------------------------------------------
