"""Generic deep research agent built on LangChain + LangGraph.

The agent takes a research question and produces a structured report by:
  1. Planning: decompose the question into focused sub-questions.
  2. Researching: for each sub-question, run a web search and summarize hits.
  3. Reflecting: identify gaps and generate follow-up queries (bounded loops).
  4. Writing: synthesize a final Markdown report with citations.

Run:
    export OPENAI_KEY=...        # LAS proxy token
    python deep_research.py "What are the tradeoffs of MoE vs dense LLMs?"
"""

from __future__ import annotations

import argparse
import json
import operator
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = os.getenv(
    "DEEP_RESEARCH_MODEL",
    "openai/gpt-5-mini",
)
DEFAULT_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "https://llm-west.ncsu-las.net/v1"
)


def build_llm(model: str = DEFAULT_MODEL, temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_KEY (LAS proxy token) before running.")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=DEFAULT_BASE_URL,
    )


# --------------------------------------------------------------------------- #
# Search tool
# --------------------------------------------------------------------------- #


DATASET_ROOT = Path(
    os.getenv(
        "DEEP_RESEARCH_DATASET",
        "/home/workshop3/efs/resources/datasets/SCADS2025/ZendiaDatasets/Clean Datasets",
    )
)

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
    """Load every JSON file under DATASET_ROOT once and index its tokens."""
    docs: list[dict[str, Any]] = []
    for path in DATASET_ROOT.rglob("*.json"):
        try:
            with path.open() as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        title = str(data.get("title", "") or "")
        body = str(data.get("body", "") or "")
        tags = data.get("tags") or []
        tag_text = " ".join(str(t) for t in tags) if isinstance(tags, list) else ""
        topic = str(data.get("topic", "") or "")
        # Title/tags/topic weighted by duplication.
        searchable = " ".join([title, title, tag_text, tag_text, topic, body])
        tokens = set(_tokenize(searchable))
        docs.append(
            {
                "path": str(path),
                "title": title,
                "body": body,
                "tags": tags,
                "tokens": tokens,
            }
        )
    return docs


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search local JSON corpus and return [{title, url, snippet}]."""
    corpus = _load_corpus()
    q_tokens = [t for t in _tokenize(query) if t not in _STOPWORDS]
    if not q_tokens:
        return []
    q_set = set(q_tokens)
    scored: list[tuple[int, dict[str, Any]]] = []
    for doc in corpus:
        score = len(q_set & doc["tokens"])
        if score > 0:
            scored.append((score, doc))
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
# State + structured outputs
# --------------------------------------------------------------------------- #


class SubQuestion(BaseModel):
    question: str = Field(description="A focused sub-question to research.")
    search_query: str = Field(description="A concise web search query for it.")


class Plan(BaseModel):
    sub_questions: list[SubQuestion] = Field(
        description="3-6 sub-questions covering the research topic."
    )


class Reflection(BaseModel):
    sufficient: bool = Field(
        description="True if the gathered notes are sufficient to write the report."
    )
    follow_ups: list[SubQuestion] = Field(
        default_factory=list,
        description="New sub-questions to fill gaps (empty if sufficient).",
    )


class Finding(TypedDict):
    question: str
    summary: str
    sources: list[dict[str, str]]


class ResearchState(TypedDict, total=False):
    topic: str
    plan: list[SubQuestion]
    pending: list[SubQuestion]
    findings: Annotated[list[Finding], operator.add]
    iterations: int
    max_iterations: int
    report: str


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

PLANNER_SYS = """You are a research planner. Given a topic, break it into 3-6 \
sub-questions that, taken together, fully cover the topic. Each sub-question \
should be answerable with a focused web search. Avoid overlap."""

RESEARCH_SYS = """You are a research analyst. You will be given a sub-question \
and a list of web search results (title, url, snippet). Write a concise, \
factual summary (4-8 sentences) that answers the sub-question using only the \
provided results. Cite sources inline as [n] referring to the result index \
(1-based). If the results are insufficient, say so explicitly."""

REFLECT_SYS = """You are a critical reviewer. Given the original topic and the \
research notes gathered so far, decide whether the notes are sufficient to \
write a thorough report. If not, propose up to 3 follow-up sub-questions that \
target the most important gaps. Be strict: only request follow-ups if there \
is a real, material gap."""

WRITER_SYS = """You are a technical writer. Synthesize the research notes into \
a well-structured Markdown report with:
  - A short executive summary.
  - Sections per major theme (not per sub-question; group related findings).
  - A "Sources" list at the end with numbered URLs.
Use inline citations like [1], [2] that map to the Sources list. Be precise, \
neutral, and avoid speculation beyond the notes."""


def plan_node(state: ResearchState) -> dict[str, Any]:
    llm = build_llm().with_structured_output(Plan)
    plan: Plan = llm.invoke(
        [
            SystemMessage(content=PLANNER_SYS),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ]
    )
    return {
        "plan": plan.sub_questions,
        "pending": list(plan.sub_questions),
        "findings": [],
        "iterations": 0,
        "max_iterations": state.get("max_iterations", 2),
    }


def research_node(state: ResearchState) -> dict[str, Any]:
    """Research every pending sub-question and append findings."""
    llm = build_llm()
    new_findings: list[Finding] = []
    for sq in state["pending"]:
        results = web_search(sq.search_query, max_results=5)
        if not results:
            new_findings.append(
                {
                    "question": sq.question,
                    "summary": "No search results returned.",
                    "sources": [],
                }
            )
            continue
        formatted = "\n".join(
            f"[{i + 1}] {r['title']}\n    {r['url']}\n    {r['snippet']}"
            for i, r in enumerate(results)
        )
        msg = llm.invoke(
            [
                SystemMessage(content=RESEARCH_SYS),
                HumanMessage(
                    content=(
                        f"Sub-question: {sq.question}\n\n"
                        f"Search results:\n{formatted}"
                    )
                ),
            ]
        )
        new_findings.append(
            {
                "question": sq.question,
                "summary": str(msg.content).strip(),
                "sources": results,
            }
        )
    return {"findings": new_findings, "pending": []}


def reflect_node(state: ResearchState) -> dict[str, Any]:
    iterations = state.get("iterations", 0) + 1
    if iterations > state.get("max_iterations", 2):
        return {"iterations": iterations, "pending": []}

    llm = build_llm().with_structured_output(Reflection)
    notes = "\n\n".join(
        f"Q: {f['question']}\nNotes: {f['summary']}" for f in state["findings"]
    )
    refl: Reflection = llm.invoke(
        [
            SystemMessage(content=REFLECT_SYS),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n\n"
                    f"Notes gathered so far:\n{notes}"
                )
            ),
        ]
    )
    pending = [] if refl.sufficient else refl.follow_ups
    return {"iterations": iterations, "pending": pending}


def should_continue(state: ResearchState) -> str:
    if state.get("pending"):
        return "research"
    return "write"


def write_node(state: ResearchState) -> dict[str, Any]:
    llm = build_llm(temperature=0.3)

    # Build a single, globally-numbered source list across all findings.
    all_sources: list[dict[str, str]] = []
    seen_urls: dict[str, int] = {}
    note_blocks: list[str] = []
    for f in state["findings"]:
        local_to_global: dict[int, int] = {}
        for i, src in enumerate(f["sources"], start=1):
            url = src["url"]
            if url not in seen_urls:
                all_sources.append(src)
                seen_urls[url] = len(all_sources)
            local_to_global[i] = seen_urls[url]
        # rewrite local [n] markers to global indices
        summary = f["summary"]
        for local, global_idx in local_to_global.items():
            summary = summary.replace(f"[{local}]", f"[[{global_idx}]]")
        # collapse double-bracket placeholder to single
        summary = summary.replace("[[", "[").replace("]]", "]")
        note_blocks.append(f"### {f['question']}\n{summary}")

    notes_md = "\n\n".join(note_blocks)
    sources_md = "\n".join(
        f"{i}. [{s['title'] or s['url']}]({s['url']})"
        for i, s in enumerate(all_sources, start=1)
    )

    msg = llm.invoke(
        [
            SystemMessage(content=WRITER_SYS),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n\n"
                    f"Research notes (citations already use global indices):\n"
                    f"{notes_md}\n\n"
                    f"Use this Sources list verbatim at the end:\n{sources_md}"
                )
            ),
        ]
    )
    return {"report": str(msg.content).strip()}


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("plan", plan_node)
    g.add_node("research", research_node)
    g.add_node("reflect", reflect_node)
    g.add_node("write", write_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "research")
    g.add_edge("research", "reflect")
    g.add_conditional_edges(
        "reflect", should_continue, {"research": "research", "write": "write"}
    )
    g.add_edge("write", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def run(topic: str, max_iterations: int = 2, verbose: bool = False) -> str:
    graph = build_graph()
    final: dict[str, Any] = {}
    for event in graph.stream(
        {"topic": topic, "max_iterations": max_iterations},
        stream_mode="updates",
    ):
        if verbose:
            for node, update in event.items():
                preview = {
                    k: (
                        f"<{len(v)} items>"
                        if isinstance(v, list)
                        else (v if not isinstance(v, str) else v[:120])
                    )
                    for k, v in update.items()
                }
                print(f"[{node}] {json.dumps(preview, default=str)}")
                if node == "plan" and update.get("plan"):
                    print("  Sub-questions:")
                    for i, sq in enumerate(update["plan"], start=1):
                        print(f"    {i}. {sq.question}")
                        print(f"       query: {sq.search_query}")
                if node == "reflect" and update.get("pending"):
                    print("  Follow-up sub-questions:")
                    for i, sq in enumerate(update["pending"], start=1):
                        print(f"    {i}. {sq.question}")
                        print(f"       query: {sq.search_query}")
        final.update(next(iter(event.values())))
    # The streamed `final` only holds the last update; re-run via invoke to
    # get the complete terminal state.
    state = graph.invoke({"topic": topic, "max_iterations": max_iterations})
    return state["report"]


def main() -> None:
    p = argparse.ArgumentParser(description="Deep research agent.")
    p.add_argument("topic", help="The research question or topic.")
    p.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum reflect/research loops (default: 2).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-o", "--output", help="Write the report to this file.")
    args = p.parse_args()

    report = run(args.topic, args.max_iterations, args.verbose)
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Wrote report to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
