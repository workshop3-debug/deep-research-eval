"""
DeepAgent Evaluation with Inspect AI
======================================
A generic deep-research agent evaluation harness. The agent works over any
document corpus (a directory of files) configured via the DEEP_RESEARCH_DATASET
environment variable. Evaluates the full DeepAgent pipeline as well as each
subagent individually:
  - research()   - read-only data gathering from the corpus
  - plan()       - structured planning / analysis
  - general()    - full-capability execution

The samples file (deepagent_samples.json) supplies the actual questions, so
this file stays domain-agnostic — swap the corpus and the samples and the same
solvers/scorers apply.

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
RESEARCH_CRITERIA = """- Faithfulness: every claim is supported by evidence drawn from the source documents.
- Relevance: the answer addresses what was actually asked.
- Coverage: uses multiple relevant sources rather than a single hit.
- Citations: cites specific files / identifiers so claims are traceable.
- Honest gaps: explicitly says when the sources are silent on something.
- Discipline: stays read-only and does not fabricate file contents."""

PLAN_CRITERIA = """- Structure: clearly organized phases or numbered steps.
- Specificity: steps are actionable, not vague aspirations.
- Coverage: addresses every dimension the prompt asks for (deliverables, dependencies, risks, etc.).
- Atomicity: sub-questions or steps are focused enough to execute independently.
- Success criteria: defines how each step is judged 'done'.
- No execution: produces a plan only, without running code or writing files."""

GENERAL_CRITERIA = """- Correctness: computed values and extracted facts match what the source documents actually contain.
- Completeness: every sub-task in the prompt is attempted.
- Faithfulness: claims are grounded in observed files, not invented.
- Format compliance: output matches the requested format (markdown table, word limits, ordering).
- Citations: when asked, references specific files / identifiers.
- Synthesis: where the prompt asks for interpretation, the synthesis is supported by the extracted data."""

DEEPAGENT_CRITERIA = """- Pipeline: the response shows evidence of research -> plan -> execution stages.
- Faithfulness: claims trace back to specific source documents.
- Structure: final output is a coherent markdown report with executive summary, thematic sections, and sources.
- Coverage: integrates findings across multiple sub-questions / sources.
- Citations: file identifiers or paths are cited inline or in a sources list.
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


PLAN_RELEVANCE_CRITERIA = """For each item in REQUIRED TOPICS, decide whether \
the RESPONSE addresses it: yes (1.0), partial (0.5), or no (0.0). For each \
item in FORBIDDEN TOPICS, decide whether the RESPONSE drifts into it (yes / no).

Compute: coverage = (sum of required scores) / (number of required topics).
Apply a penalty of 0.2 per forbidden topic the response drifts into.
final_fraction = max(0.0, min(1.0, coverage - penalty))

Then map final_fraction onto a 1-5 integer score:
  1 = 0.0-0.19, 2 = 0.2-0.39, 3 = 0.4-0.59, 4 = 0.6-0.79, 5 = 0.8-1.0
Return the integer score and a 1-3 sentence rationale that names which required
topics were missed and which forbidden topics (if any) were touched."""


@scorer(metrics=[mean(), stderr()])
def score_plan_relevance() -> Scorer:
    """Sample-specific topic-coverage scorer.

    Reads expected_topics (required) and forbidden_topics (optional decoys)
    from the sample's metadata and asks the judge to grade the plan's coverage.
    Samples without expected_topics are skipped (value=0.0, noted in rationale).
    """
    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        required = meta.get("expected_topics") or []
        forbidden = meta.get("forbidden_topics") or []
        if not required:
            return Score(
                value=0.0,
                explanation="sample has no expected_topics; relevance not graded",
                metadata={"skipped": True},
            )

        response = state.output.completion or ""
        required_block = "REQUIRED TOPICS:\n" + "\n".join(f"- {t}" for t in required)
        forbidden_block = (
            "\n\nFORBIDDEN TOPICS:\n" + "\n".join(f"- {t}" for t in forbidden)
            if forbidden else ""
        )
        material = (
            f"{required_block}{forbidden_block}\n\n"
            f"PLAN:\n{response or '(empty)'}"
        )
        target_text = target.text if hasattr(target, "text") else str(target)
        s, rationale = await _judge(
            PLAN_RELEVANCE_CRITERIA, state.input_text, target_text, material
        )
        return Score(
            value=s,
            explanation=rationale,
            metadata={"required": required, "forbidden": forbidden},
        )

    return score


@scorer(metrics=[mean(), stderr()])
def score_general_rubric() -> Scorer:
    return _rubric_scorer(GENERAL_CRITERIA, "general")


@scorer(metrics=[mean(), stderr()])
def score_deepagent_rubric() -> Scorer:
    return _rubric_scorer(DEEPAGENT_CRITERIA, "deepagent")


# Alias: the relevance scorer is generic and works on any response, not just
# plans. Use this name when wiring it into research tasks.
score_topic_relevance = score_plan_relevance


@scorer(metrics=[mean(), stderr()])
def score_required_citations() -> Scorer:
    """Reference-preservation check.

    Reads sample metadata.required_citations (list of serial strings or file
    basenames) and verifies each appears at least once in the response.
    Score = matches / required. Samples without required_citations are skipped.
    """
    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        required = [str(c) for c in (meta.get("required_citations") or [])]
        if not required:
            return Score(
                value=0.0,
                explanation="sample has no required_citations; not graded",
                metadata={"skipped": True},
            )
        response = (state.output.completion or "").lower()
        hits, misses = [], []
        for cite in required:
            (hits if cite.lower() in response else misses).append(cite)
        value = len(hits) / len(required)
        return Score(
            value=value,
            explanation=(
                f"{len(hits)}/{len(required)} required citations present"
                + (f"; missing: {', '.join(misses)}" if misses else "")
            ),
            metadata={"hits": hits, "missing": misses},
        )

    return score


# ---------------------------------------------------------------------------
# Research-summary faithfulness scorers
# ---------------------------------------------------------------------------

_FETCHED_TOOLS = {"read_file", "grep"}


def _collected_evidence(state: TaskState, max_chars: int = 60_000) -> str:
    """Concatenate everything the agent fetched via read_file / grep tool calls."""
    chunks: list[str] = []
    for msg in state.messages:
        if getattr(msg, "role", None) != "tool":
            continue
        if getattr(msg, "function", None) not in _FETCHED_TOOLS:
            continue
        text = msg.text or ""
        if not text.strip():
            continue
        chunks.append(f"--- tool={msg.function} ---\n{text}")
    blob = "\n\n".join(chunks)
    if len(blob) > max_chars:
        blob = blob[:max_chars] + "\n…[evidence truncated]"
    return blob


FAITHFULNESS_CRITERIA = """Evaluate whether the SUMMARY is grounded in the \
EVIDENCE (the actual tool results the agent fetched from the corpus).
- Faithfulness: every claim in the summary appears in or is a reasonable paraphrase of the evidence.
- No fabrication: no facts, names, dates, or numbers are introduced that are absent from the evidence.
- Coverage: the summary actually uses what was fetched rather than ignoring it.
- Honest gaps: where the evidence is silent, the summary says so explicitly.
- Calibration: confident claims are well-supported; speculative claims are flagged."""


@scorer(metrics=[mean(), stderr()])
def score_research_faithfulness() -> Scorer:
    """Approach A: tool-trace faithfulness.

    Collects everything the agent read via read_file / grep and judges the
    summary against that concatenated evidence. Catches hallucinated content
    even when no citation is given.
    """
    async def score(state: TaskState, target: Target) -> Score:
        evidence = _collected_evidence(state)
        if not evidence:
            return Score(
                value=0.0,
                explanation="agent fetched no documents via read_file/grep",
                metadata={"evidence_chars": 0},
            )
        response = state.output.completion or ""
        material = (
            f"EVIDENCE (tool results, may be truncated):\n{evidence}\n\n"
            f"SUMMARY:\n{response or '(empty)'}"
        )
        target_text = target.text if hasattr(target, "text") else str(target)
        s, rationale = await _judge(
            FAITHFULNESS_CRITERIA, state.input_text, target_text, material
        )
        return Score(
            value=s,
            explanation=rationale,
            metadata={"evidence_chars": len(evidence)},
        )

    return score


# Citation patterns: 5-digit serials (Zendia convention), file:// URIs, and
# bare *.json paths. Adjust _CITE_RE if your corpus uses different identifiers.
_CITE_RE = re.compile(
    r"file://(?P<uri>[^\s\]\)\"'`]+)"
    r"|(?P<json>[\w./-]+\.json)"
    r"|(?:serial[:= ]+)?(?P<serial>\d{5})\b"
)


def _extract_citations(text: str) -> list[str]:
    seen: list[str] = []
    for m in _CITE_RE.finditer(text or ""):
        token = m.group("uri") or m.group("json") or m.group("serial")
        if token and token not in seen:
            seen.append(token)
    return seen


def _resolve_citation(token: str, dataset_root: Path) -> Path | None:
    """Map a citation token to an on-disk file under dataset_root."""
    # file:// URI - strip the scheme.
    if token.startswith("/"):
        p = Path(token)
        return p if p.is_file() else None
    # Direct relative path under the dataset root.
    direct = dataset_root / token
    if direct.is_file():
        return direct
    # Serial form (e.g. "00152") -> search for matching filename.
    if token.isdigit() and dataset_root.is_dir():
        for match in dataset_root.rglob(f"{token}.json"):
            return match
    # Last resort: glob by basename.
    if dataset_root.is_dir():
        for match in dataset_root.rglob(Path(token).name):
            return match
    return None


CITATION_CRITERIA = """You will see a CITED SOURCE (the file the summary cites) \
and the SUMMARY itself. Decide whether the claims in the summary that are \
attached to this citation are actually supported by the cited source.
- Direct support: the cited text says what the summary claims.
- Paraphrase: the cited text supports the claim in different words.
- Unsupported: the claim is not present in the cited source.
- Contradicted: the cited source says something different.
Score on the 1-5 scale: 5 = fully supported, 3 = partially supported, 1 = unsupported or contradicted."""


@scorer(metrics=[mean(), stderr()])
def score_research_citations() -> Scorer:
    """Approach B: per-citation verification.

    Parses citations out of the summary, reads each cited file from disk, and
    judges each citation independently. Final score = mean of per-citation
    scores. Returns 0.0 with an explanation if no citations are found.
    """
    dataset_root = Path(DATASET_PATH)

    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion or ""
        citations = _extract_citations(response)
        if not citations:
            return Score(
                value=0.0,
                explanation="summary contains no resolvable citations",
                metadata={"citations": []},
            )

        per_citation: list[dict[str, Any]] = []
        scores: list[float] = []
        target_text = target.text if hasattr(target, "text") else str(target)

        for token in citations:
            path = _resolve_citation(token, dataset_root)
            if path is None:
                per_citation.append({"citation": token, "score": 0.0,
                                     "note": "unresolved"})
                scores.append(0.0)
                continue
            try:
                cited_text = path.read_text(errors="replace")
            except OSError as exc:
                per_citation.append({"citation": token, "score": 0.0,
                                     "note": f"read error: {exc}"})
                scores.append(0.0)
                continue
            if len(cited_text) > 20_000:
                cited_text = cited_text[:20_000] + "\n…[source truncated]"
            material = (
                f"CITATION TOKEN: {token}\n"
                f"CITED SOURCE ({path.name}):\n{cited_text}\n\n"
                f"SUMMARY:\n{response}"
            )
            s, rationale = await _judge(
                CITATION_CRITERIA, state.input_text, target_text, material
            )
            per_citation.append({
                "citation": token,
                "path": str(path),
                "score": s,
                "rationale": rationale,
            })
            scores.append(s)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        unresolved = sum(1 for c in per_citation if c.get("note") == "unresolved")
        explanation = (
            f"{len(scores)} citations checked; "
            f"{unresolved} unresolved; mean={mean_score:.2f}"
        )
        return Score(
            value=mean_score,
            explanation=explanation,
            metadata={"citations": per_citation},
        )

    return score


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


# # ---------------------------------------------------------------------------
# # Task 1 - Research subagent (standalone)
# # ---------------------------------------------------------------------------

# @task
# def task_research_agent() -> Task:
#     """Evaluate the research() subagent on corpus extraction tasks."""
#     research_subagent = research(
#         tools=READONLY_TOOLS,
#         instructions=dedent(f"""\
#             You have read-only access to a document corpus at:
#             {DATASET_PATH}

#             Use list_files, read_file, and grep to gather information from the
#             corpus. Cite specific file paths or identifiers when making claims.
#             Never attempt to write or modify files.
#         """),
#     )

#     agent = deepagent(
#         subagents=[research_subagent],
#         memory=False,
#         instructions=(
#             f"Delegate ALL work to the research subagent. Dataset: {DATASET_PATH}"
#         ),
#     )

#     return Task(
#         dataset=load_samples("research"),
#         solver=as_solver(agent),
#         scorer=[
#             task_completion_scorer(),
#             score_research_rubric(),
#             score_research_faithfulness(),
#             score_research_citations(),
#             score_topic_relevance(),
#             score_required_citations(),
#         ],
#         metadata={"subagent": "research"},
#         sandbox="local",
#     )


# # ---------------------------------------------------------------------------
# # Task 2 - Plan subagent (standalone)
# # ---------------------------------------------------------------------------

# @task
# def task_plan_agent() -> Task:
#     """Evaluate the plan() subagent on research-planning tasks."""
#     plan_subagent = plan(
#         tools=READONLY_TOOLS,
#         instructions=dedent(f"""\
#             You are a research planning specialist.
#             The corpus you may inspect (read-only) lives at:
#             {DATASET_PATH}

#             Produce structured, actionable research plans. Break the task into
#             focused sub-questions, list the search terms and document fields
#             you would use, and state clear success criteria for each step. Do
#             not execute code, modify files, or produce the final analysis.
#         """),
#     )

#     agent = deepagent(
#         subagents=[plan_subagent],
#         memory=False,
#         instructions=(
#             f"Delegate ALL planning work to the plan subagent. Dataset: {DATASET_PATH}"
#         ),
#     )

#     return Task(
#         dataset=load_samples("plan"),
#         solver=as_solver(agent),
#         scorer=[
#             task_completion_scorer(),
#             score_plan_rubric(),
#             score_plan_relevance(),
#         ],
#         metadata={"subagent": "plan"},
#         sandbox="local",
#     )


# # ---------------------------------------------------------------------------
# # Task 3 - General subagent (standalone)
# # ---------------------------------------------------------------------------

# @task
# def task_general_agent() -> Task:
#     """Evaluate the general() subagent on corpus-wide computations."""
#     general_subagent = general(
#         tools=READWRITE_TOOLS,
#         instructions=dedent(f"""\
#             You are a capable general-purpose analyst.
#             Corpus directory: {DATASET_PATH}

#             You may read AND process files (bash, text_editor). Use Python
#             via bash for aggregation and statistics. Cite specific file paths
#             or identifiers when summarizing findings.
#         """),
#         memory="readwrite",
#     )

#     agent = deepagent(
#         subagents=[general_subagent],
#         memory=True,
#         instructions=(
#             f"Delegate ALL execution work to the general subagent. Dataset: {DATASET_PATH}"
#         ),
#     )

#     return Task(
#         dataset=load_samples("general"),
#         solver=as_solver(agent),
#         scorer=[task_completion_scorer(), score_general_rubric()],
#         metadata={"subagent": "general"},
#         sandbox="local",
#     )


# ---------------------------------------------------------------------------
# Task 4 - Full DeepAgent (all three subagents collaborating)
# ---------------------------------------------------------------------------

@task
def task_deepagent_full() -> Task:
    """Evaluate the complete DeepAgent: research -> plan -> general."""
    research_subagent = research(
        tools=READONLY_TOOLS,
        instructions=dedent(f"""\
            Your role: read-only exploration of the document corpus.
            Corpus path: {DATASET_PATH}

            Use list_files, read_file, and grep. Summarise findings clearly
            for the planning stage, citing specific file paths or identifiers.
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

            Use bash for aggregations. Produce a markdown report with an
            executive summary, thematic sections, and a Sources list citing
            specific file paths or identifiers.
        """),
        memory="readwrite",
    )

    agent = deepagent(
        subagents=[research_subagent, plan_subagent, general_subagent],
        memory=True,
        todo_write=True,
        max_depth=2,
        instructions=dedent(f"""\
            You orchestrate three specialised subagents over a document corpus:
              - research  : read-only data exploration
              - plan      : structured planning
              - general   : code execution and report writing

            Corpus path: {DATASET_PATH}

            Always follow the sequence: research -> plan -> general.
            Consolidate outputs into a final markdown report with an executive
            summary, thematic sections, and a Sources list of file references.
        """),
    )

    return Task(
        dataset=load_samples("deepagent"),
        solver=as_solver(agent),
        scorer=[
            task_completion_scorer(),
            score_deepagent_rubric(),
            score_research_faithfulness(),
            score_research_citations(),
            score_topic_relevance(),
            score_required_citations(),
        ],
        metadata={"subagent": "deepagent_full"},
        sandbox="local",
    )


# ---------------------------------------------------------------------------
# To run ALL tasks in one sweep, just point inspect at this file with no
# task name - it will auto-discover all @task functions:
#
#   inspect eval deep_agent_inspect.py --model openai/gpt-5-mini
# ---------------------------------------------------------------------------
