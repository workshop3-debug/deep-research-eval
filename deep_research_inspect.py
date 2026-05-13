"""Deep research agent built with inspect-ai.

Pipeline (each step is a solver that reads/writes state.store):
  1. plan       — decompose the question into 3-6 sub-questions
  2. research   — for each pending sub-question, search the local JSON corpus
                  and summarize the top hits
  3. reflect    — decide whether the notes are sufficient; if not, propose
                  follow-up sub-questions (bounded by MAX_ITERATIONS)
  4. write      — synthesize a Markdown report with inline citations

Each phase is also graded by an LLM-judge scorer:
  - score_plan, score_research, score_reflection, score_summary

Run:
  export OPENAI_KEY=...                       # LAS proxy token
  inspect eval deep_research_inspect.py --model openai/gpt-5-mini
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, chain, solver

# --------------------------------------------------------------------------- #
# LLM proxy wiring — LAS proxy speaks the OpenAI protocol.
# --------------------------------------------------------------------------- #

if os.getenv("OPENAI_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]
os.environ.setdefault("OPENAI_BASE_URL", "https://llm-west.ncsu-las.net/v1")

MAX_ITERATIONS = int(os.getenv("DEEP_RESEARCH_MAX_ITERS", "2"))
MAX_RESULTS = int(os.getenv("DEEP_RESEARCH_MAX_RESULTS", "5"))
DATASET_ROOT = Path(
    os.getenv(
        "DEEP_RESEARCH_DATASET",
        "/home/workshop3/efs/resources/datasets/SCADS2025/ZendiaDatasets/Clean Datasets",
    )
)


# --------------------------------------------------------------------------- #
# Local JSON corpus search
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "by", "with", "as", "at", "from", "that", "this",
    "it", "its", "what", "which", "who", "whom", "how", "why", "when", "where",
    "do", "does", "did", "about", "into", "their", "they", "them", "we", "our",
}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@lru_cache(maxsize=1)
def _load_corpus() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    if not DATASET_ROOT.exists():
        return docs
    for path in DATASET_ROOT.rglob("*.json"):
        try:
            with path.open() as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        title = str(data.get("title", "") or "")
        body = str(data.get("body", "") or "")
        tags = data.get("tags") or []
        tag_text = " ".join(str(t) for t in tags) if isinstance(tags, list) else ""
        topic = str(data.get("topic", "") or "")
        searchable = " ".join([title, title, tag_text, tag_text, topic, body])
        docs.append(
            {
                "path": str(path),
                "title": title,
                "body": body,
                "tokens": set(_tokenize(searchable)),
            }
        )
    return docs


def search_corpus(query: str, max_results: int = MAX_RESULTS) -> list[dict[str, str]]:
    corpus = _load_corpus()
    q_set = {t for t in _tokenize(query) if t not in _STOPWORDS}
    if not q_set or not corpus:
        return []
    scored = [(len(q_set & doc["tokens"]), doc) for doc in corpus]
    scored = [(s, d) for s, d in scored if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, str]] = []
    for _, doc in scored[:max_results]:
        body = doc["body"]
        snippet = body[:400] + ("..." if len(body) > 400 else "")
        out.append(
            {
                "title": doc["title"],
                "url": f"file://{doc['path']}",
                "snippet": snippet,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def _parse_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM response."""
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


async def _ask_json(system: str, user: str, fallback: Any) -> Any:
    out = await get_model().generate(
        [ChatMessageSystem(content=system), ChatMessageUser(content=user)],
        config=GenerateConfig(temperature=0.2),
    )
    parsed = _parse_json(out.completion)
    return parsed if parsed is not None else fallback


# --------------------------------------------------------------------------- #
# Solver: plan
# --------------------------------------------------------------------------- #

PLANNER_SYS = """You are a research planner. Given a topic, break it into 3-6 \
sub-questions that, taken together, fully cover the topic. Each sub-question \
should be answerable with a focused document search. Avoid overlap.

Respond with valid JSON only:
{"sub_questions": [
  {"question": "...", "search_query": "..."},
  ...
]}"""


@solver
def plan_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        topic = state.input_text
        data = await _ask_json(
            PLANNER_SYS,
            f"Topic: {topic}",
            fallback={"sub_questions": [{"question": topic, "search_query": topic}]},
        )
        sub_qs = data.get("sub_questions", []) if isinstance(data, dict) else []
        sub_qs = [
            sq for sq in sub_qs
            if isinstance(sq, dict) and sq.get("question") and sq.get("search_query")
        ]
        state.store.set("topic", topic)
        state.store.set("plan", sub_qs)
        state.store.set("pending", list(sub_qs))
        state.store.set("findings", [])
        state.store.set("iterations", 0)
        return state

    return solve


# --------------------------------------------------------------------------- #
# Solver: research (one pass over pending sub-questions)
# --------------------------------------------------------------------------- #

RESEARCH_SYS = """You are a research analyst. You will be given a sub-question \
and a list of document search results (title, url, snippet). Write a concise, \
factual summary (4-8 sentences) that answers the sub-question using only the \
provided results. Cite sources inline as [n] referring to the result index \
(1-based). If the results are insufficient, say so explicitly."""


@solver
def research_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        pending = state.store.get("pending", []) or []
        findings = state.store.get("findings", []) or []
        model = get_model()
        for sq in pending:
            results = search_corpus(sq["search_query"])
            if not results:
                findings.append(
                    {
                        "question": sq["question"],
                        "summary": "No search results returned.",
                        "sources": [],
                    }
                )
                continue
            formatted = "\n".join(
                f"[{i + 1}] {r['title']}\n    {r['url']}\n    {r['snippet']}"
                for i, r in enumerate(results)
            )
            out = await model.generate(
                [
                    ChatMessageSystem(content=RESEARCH_SYS),
                    ChatMessageUser(
                        content=(
                            f"Sub-question: {sq['question']}\n\n"
                            f"Search results:\n{formatted}"
                        )
                    ),
                ],
                config=GenerateConfig(temperature=0.2),
            )
            findings.append(
                {
                    "question": sq["question"],
                    "summary": out.completion.strip(),
                    "sources": results,
                }
            )
        state.store.set("findings", findings)
        state.store.set("pending", [])
        return state

    return solve


# --------------------------------------------------------------------------- #
# Solver: reflect (may enqueue follow-up sub-questions)
# --------------------------------------------------------------------------- #

REFLECT_SYS = """You are a critical reviewer. Given the original topic and the \
research notes gathered so far, decide whether the notes are sufficient to \
write a thorough report. If not, propose up to 3 follow-up sub-questions that \
target the most important gaps. Be strict: only request follow-ups if there \
is a real, material gap.

Respond with valid JSON only:
{
  "sufficient": true|false,
  "follow_ups": [
    {"question": "...", "search_query": "..."}
  ]
}"""


@solver
def reflect_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        iterations = state.store.get("iterations", 0)
        if iterations >= MAX_ITERATIONS:
            state.store.set("pending", [])
            return state

        topic = state.store.get("topic", state.input_text)
        findings = state.store.get("findings", []) or []
        notes = "\n\n".join(
            f"Q: {f['question']}\nNotes: {f['summary']}" for f in findings
        )
        data = await _ask_json(
            REFLECT_SYS,
            f"Topic: {topic}\n\nNotes gathered so far:\n{notes}",
            fallback={"sufficient": True, "follow_ups": []},
        )
        sufficient = bool(data.get("sufficient", True)) if isinstance(data, dict) else True
        follow_ups = data.get("follow_ups", []) if isinstance(data, dict) else []
        follow_ups = [
            sq for sq in follow_ups
            if isinstance(sq, dict) and sq.get("question") and sq.get("search_query")
        ]
        state.store.set("iterations", iterations + 1)
        state.store.set("sufficient", sufficient)
        reflections = state.store.get("reflections", []) or []
        reflections.append({"sufficient": sufficient, "follow_ups": follow_ups})
        state.store.set("reflections", reflections)
        state.store.set("pending", [] if sufficient else follow_ups)
        return state

    return solve


# --------------------------------------------------------------------------- #
# Solver: research/reflect loop wrapper
# --------------------------------------------------------------------------- #


@solver
def research_loop() -> Solver:
    """Run research, then reflect, looping while reflect produces follow-ups."""
    research = research_solver()
    reflect = reflect_solver()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await research(state, generate)
        for _ in range(MAX_ITERATIONS):
            state = await reflect(state, generate)
            if not state.store.get("pending"):
                break
            state = await research(state, generate)
        return state

    return solve


# --------------------------------------------------------------------------- #
# Solver: write final report
# --------------------------------------------------------------------------- #

WRITER_SYS = """You are a technical writer. Synthesize the research notes into \
a well-structured Markdown report with:
  - A short executive summary.
  - Sections per major theme (not per sub-question; group related findings).
  - A "Sources" list at the end with numbered URLs.
Use inline citations like [1], [2] that map to the Sources list. Be precise, \
neutral, and avoid speculation beyond the notes."""


@solver
def write_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        topic = state.store.get("topic", state.input_text)
        findings = state.store.get("findings", []) or []

        all_sources: list[dict[str, str]] = []
        seen: dict[str, int] = {}
        note_blocks: list[str] = []
        for f in findings:
            local_to_global: dict[int, int] = {}
            for i, src in enumerate(f.get("sources", []), start=1):
                url = src["url"]
                if url not in seen:
                    all_sources.append(src)
                    seen[url] = len(all_sources)
                local_to_global[i] = seen[url]
            summary = f["summary"]
            for local, gidx in local_to_global.items():
                summary = summary.replace(f"[{local}]", f"[[{gidx}]]")
            summary = summary.replace("[[", "[").replace("]]", "]")
            note_blocks.append(f"### {f['question']}\n{summary}")

        notes_md = "\n\n".join(note_blocks) or "(no findings)"
        sources_md = "\n".join(
            f"{i}. [{s['title'] or s['url']}]({s['url']})"
            for i, s in enumerate(all_sources, start=1)
        ) or "(none)"

        out = await get_model().generate(
            [
                ChatMessageSystem(content=WRITER_SYS),
                ChatMessageUser(
                    content=(
                        f"Topic: {topic}\n\n"
                        f"Research notes (citations already use global indices):\n"
                        f"{notes_md}\n\n"
                        f"Use this Sources list verbatim at the end:\n{sources_md}"
                    )
                ),
            ],
            config=GenerateConfig(temperature=0.3),
        )
        report = out.completion.strip()
        state.store.set("report", report)
        state.output.completion = report
        return state

    return solve


# --------------------------------------------------------------------------- #
# LLM-judge scorers (one per pipeline phase)
# --------------------------------------------------------------------------- #

JUDGE_TEMPLATE = """You are a strict evaluator. Score the following on a 1-5 \
scale and respond with valid JSON only:
{{"score": <1-5 integer>, "rationale": "<1-3 sentences>"}}

Criteria:
{criteria}

Original question:
{topic}

Material to evaluate:
{material}
"""


async def _judge(criteria: str, topic: str, material: str) -> tuple[float, str]:
    data = await _ask_json(
        "You are a strict evaluator. Respond with valid JSON only.",
        JUDGE_TEMPLATE.format(criteria=criteria, topic=topic, material=material),
        fallback={"score": 0, "rationale": "judge failed to return JSON"},
    )
    try:
        raw = float(data.get("score", 0))
    except (TypeError, ValueError):
        raw = 0.0
    score = max(0.0, min(5.0, raw)) / 5.0  # normalize to [0,1]
    return score, str(data.get("rationale", ""))


PLAN_CRITERIA = """- Relevance: sub-questions directly serve the topic.
- Coverage: collectively cover all major dimensions.
- Non-redundancy: sub-questions are distinct.
- Atomicity: each sub-question is focused enough for one round of search.
- Answerability: each is plausibly answerable from a document search.
- Appropriate scope: not too shallow, not over-decomposed."""

RESEARCH_CRITERIA = """- Faithfulness: every claim is supported by the cited sources.
- Relevance: the summary actually answers the sub-question.
- Coverage: uses the available sources rather than ignoring them.
- Conciseness: 4-8 focused sentences, no padding.
- Honest gaps: explicitly says when sources are insufficient."""

REFLECT_CRITERIA = """- Judgment: the 'sufficient' decision is well-calibrated given the notes.
- Gap identification: follow-ups (if any) target real, material gaps.
- Non-redundancy: follow-ups don't duplicate prior sub-questions.
- Restraint: doesn't invent follow-ups when notes already cover the topic."""

SUMMARY_CRITERIA = """- Faithfulness: every claim traces back to the research notes.
- Structure: executive summary + thematic sections + sources list.
- Citations: inline [n] markers map to numbered sources.
- Coverage: integrates findings from across the sub-questions.
- Clarity: precise, neutral prose; no speculation beyond the notes."""


@scorer(metrics=[mean(), stderr()])
def score_plan() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        plan = state.store.get("plan", []) or []
        topic = state.store.get("topic", state.input_text)
        material = json.dumps(plan, indent=2) if plan else "(empty plan)"
        s, rationale = await _judge(PLAN_CRITERIA, topic, material)
        return Score(value=s, explanation=rationale, metadata={"plan": plan})

    return score


@scorer(metrics=[mean(), stderr()])
def score_research() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        findings = state.store.get("findings", []) or []
        topic = state.store.get("topic", state.input_text)
        if not findings:
            return Score(value=0.0, explanation="no findings produced")
        blocks = []
        for f in findings:
            srcs = "\n".join(
                f"  [{i + 1}] {s['title']} — {s['url']}"
                for i, s in enumerate(f.get("sources", []))
            ) or "  (no sources)"
            blocks.append(
                f"Q: {f['question']}\nSummary: {f['summary']}\nSources:\n{srcs}"
            )
        material = "\n\n".join(blocks)
        s, rationale = await _judge(RESEARCH_CRITERIA, topic, material)
        return Score(value=s, explanation=rationale)

    return score


@scorer(metrics=[mean(), stderr()])
def score_reflection() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        reflections = state.store.get("reflections", []) or []
        findings = state.store.get("findings", []) or []
        topic = state.store.get("topic", state.input_text)
        if not reflections:
            return Score(value=0.0, explanation="reflect phase did not run")
        notes = "\n".join(f"- {f['question']}" for f in findings)
        material = (
            f"Sub-questions researched:\n{notes}\n\n"
            f"Reflection log:\n{json.dumps(reflections, indent=2)}"
        )
        s, rationale = await _judge(REFLECT_CRITERIA, topic, material)
        return Score(value=s, explanation=rationale)

    return score


@scorer(metrics=[mean(), stderr()])
def score_summary() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        report = state.store.get("report") or state.output.completion or ""
        topic = state.store.get("topic", state.input_text)
        findings = state.store.get("findings", []) or []
        notes_md = "\n\n".join(
            f"Q: {f['question']}\nNotes: {f['summary']}" for f in findings
        ) or "(no notes)"
        material = (
            f"Research notes:\n{notes_md}\n\n"
            f"Final report:\n{report or '(empty)'}"
        )
        s, rationale = await _judge(SUMMARY_CRITERIA, topic, material)
        return Score(value=s, explanation=rationale)

    return score


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #


@task
def deep_research() -> Task:
    return Task(
        dataset=json_dataset("zendia_questions.json"),
        solver=chain(
            plan_solver(),
            research_loop(),
            write_solver(),
        ),
        scorer=[score_plan(), score_research(), score_reflection(), score_summary()],
    )
