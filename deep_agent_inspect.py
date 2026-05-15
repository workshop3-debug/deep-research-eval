"""
DeepAgent Evaluation with Inspect AI
======================================
Evaluates the full DeepAgent pipeline as well as each subagent individually:
  - plan()       - structured task decomposition and research planning
  - research()   - read-only targeted information gathering from the SCADS2025 dataset
  - general()    - full-capability report synthesis and citation

Scorers
-------
  - task_completion_scorer    - LLM judge: C/P/I → 1.0/0.5/0.0 overall task completion
  - retrieval_accuracy_scorer - LLM judge: did research find all expected documents?
  - faithfulness_scorer       - LLM judge: are all claims attributed to cited serials?
  - executive_summary_scorer  - Non-scoring: generates a 3-5 sentence executive summary
                                of the agent output; stored in Score.explanation for
                                review in the Inspect UI. Always returns value=1.0.

Pipeline order: plan → research → general

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

# import litellm
# litellm.modify_params = True

from inspect_ai import Task, task
from inspect_ai.agent import as_solver, deepagent, general, plan, research
from inspect_ai.dataset import Sample
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    model_graded_fact,
    model_graded_qa,
    scorer,
)
from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.solver import TaskState
from inspect_ai.tool import bash, grep, list_files, read_file, text_editor
from inspect_ai.util._limit import message_limit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLES_FILE = Path(__file__).parent / "deepagent_samples.json"

DATASET_PATH = "/home/workshop11/deep_agent_eval/1_5_3_GPT 4"

AGENT_MODEL = "openai/deepinfra/anthropic/claude-4-sonnet"

JUDGE_MODEL = "openai/gpt-4o-mini"

# How many model turns the research subagent may take per invocation.
# Reading ~127 JSON files with relevance filtering can consume 150-250 messages;
# 300 gives headroom for thorough document-by-document retrieval.
RESEARCH_MESSAGE_LIMIT = 300

# ---------------------------------------------------------------------------
# Load samples from JSON
# ---------------------------------------------------------------------------

def load_samples(role: str) -> list[Sample]:
    """
    Read deepagent_samples.json and return samples matching `role`.
    Each JSON object must have: id, input, target, and a top-level 'role' key.

    Any occurrence of the literal string ``{DATASET_PATH}`` in the input,
    target, or metadata values is replaced with the value of DATASET_PATH
    defined above.  This allows the JSON file to be kept path-agnostic while
    still resolving to the correct runtime path.
    """
    with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)

    def _sub(value: Any) -> Any:
        """Recursively substitute {DATASET_PATH} in strings and dicts."""
        if isinstance(value, str):
            return value.replace("{DATASET_PATH}", DATASET_PATH)
        if isinstance(value, dict):
            return {k: _sub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sub(v) for v in value]
        return value

    return [
        Sample(
            id=entry["id"],
            input=_sub(entry["input"]),
            target=_sub(entry["target"]),
            metadata=_sub(entry.get("metadata", {})),
        )
        for entry in raw
        if entry.get("role") == role
    ]


# ---------------------------------------------------------------------------
# Custom scorer: task-completion rubric via LLM judge
# ---------------------------------------------------------------------------

def task_completion_scorer(model: str = JUDGE_MODEL) -> Scorer:
    """
    LLM-as-judge: rates overall task completion on a 0 / 0.5 / 1.0 scale.
      C (correct / complete)  -> 1.0
      P (partial)             -> 0.5
      I (incorrect / missing) -> 0.0
    """
    base = model_graded_qa(
        model=model,
        partial_credit=True,
        instructions=dedent("""
            You are an expert evaluator.

            Rate the ASSISTANT RESPONSE against the CRITERION below.

            Scoring rubric:
              C (correct / complete)   - fully satisfies the criterion
              P (partial)              - partially satisfies, some gaps
              I (incorrect/missing)    - does not satisfy the criterion

            Think step by step, then end your response with exactly one of:
            GRADE: C
            GRADE: P
            GRADE: I
        """),
    )

    @scorer(metrics=[accuracy()])
    def task_completion() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return await base(state, target)
        return score

    return task_completion()


def retrieval_accuracy_scorer(model: str = JUDGE_MODEL) -> Scorer:
    """
    LLM-as-judge for retrieval completeness.

    The CRITERION (target) lists the expected document serial numbers that the
    research agent must retrieve.  The scorer checks how many of those appear
    in the agent's response — citation by serial number is the signal.

      C -> all or nearly all expected serials are cited and have key findings
      P -> majority of expected serials cited, but some are missing
      I -> fewer than half cited, or critical documents absent
    """
    base = model_graded_qa(
        model=model,
        partial_credit=True,
        instructions=dedent("""
            You are evaluating whether a research agent correctly retrieved
            all expected intelligence documents from a dataset.

            The CRITERION lists the expected document serial numbers (e.g.
            00006, 00041, 00042 …) that must appear in the response.

            The ASSISTANT RESPONSE contains the agent's retrieved documents.

            Evaluation steps:
              1. Extract every serial number cited in the ASSISTANT RESPONSE.
              2. Compare against the expected serials in the CRITERION.
              3. Note which expected serials are present and which are absent.

            Scoring:
              C - All or nearly all (≥90%) expected serials are cited with at
                  least one key finding each.
              P - More than half of expected serials are cited, but notable
                  gaps remain.
              I - Fewer than half of expected serials are cited, or the most
                  critical documents are missing.

            Think step by step, then end with exactly one of:
            GRADE: C
            GRADE: P
            GRADE: I
        """),
    )

    @scorer(metrics=[accuracy()])
    def retrieval_accuracy() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return await base(state, target)
        return score

    return retrieval_accuracy()


def faithfulness_scorer(model: str = JUDGE_MODEL) -> Scorer:
    """
    LLM-as-judge for source faithfulness.

    Checks that the generated intelligence report does not contain fabricated
    claims beyond what the retrieved source documents state.

      C -> all significant claims are attributed to a cited serial; no obvious
           fabrication detected
      P -> most claims sourced; a few unsourced specifics present
      I -> multiple fabricated facts, invented names/figures, or material
           claims with no serial citation
    """
    base = model_graded_qa(
        model=model,
        partial_credit=True,
        instructions=dedent("""
            You are evaluating whether an intelligence report is faithful to
            its stated source documents.

            The CRITERION describes what faithful grounding means for this
            specific task (which serials to cite, what must not be fabricated).

            The ASSISTANT RESPONSE is the generated intelligence report.

            Evaluation steps:
              1. Identify every specific factual claim (names, dates, figures,
                 technical details, locations) in the ASSISTANT RESPONSE.
              2. Check whether each claim is attributed to a cited serial
                 number in the format (serial: XXXXX) or equivalent citation.
              3. Flag any claim that introduces specific details not present
                 in the stated sources — these are fabrication signals.
              4. For hallucination-trap tasks (fictional entity / fictional
                 person / leading premise): check that the agent correctly
                 declined to confirm or fabricate the requested fiction.

            Scoring:
              C - All significant claims have serial citations; no apparent
                  fabrication beyond what the sources state; hallucination
                  traps correctly refused.
              P - Most claims are sourced, but some unsourced specifics appear
                  that cannot be verified from citations alone.
              I - Multiple uncited specifics that appear invented; or the agent
                  confirmed a hallucination trap by fabricating requested
                  details.

            Think step by step, then end with exactly one of:
            GRADE: C
            GRADE: P
            GRADE: I
        """),
    )

    @scorer(metrics=[accuracy()])
    def faithfulness() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return await base(state, target)
        return score

    return faithfulness()


def executive_summary_scorer(model: str = JUDGE_MODEL) -> Scorer:
    """
    Generates a concise executive summary of the deep agent's output.
    Does not produce a meaningful score (always returns 1.0); the summary
    is stored in Score.explanation for review in the Inspect UI.
    """
    judge = get_model(model)

    @scorer(metrics=[accuracy()])
    def executive_summary() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            output = state.output.completion or ""
            prompt = dedent(f"""\
                You are an intelligence analyst reviewing an AI-generated report.

                Write a 3–5 sentence executive summary of the ASSISTANT RESPONSE
                below. Capture: the main intelligence question addressed, the key
                findings or conclusions, and any significant caveats or gaps noted.

                ASSISTANT RESPONSE:
                {output}

                Executive summary:
            """)
            result = await judge.generate([ChatMessageUser(content=prompt)])
            summary = result.completion.strip()
            return Score(value=1.0, explanation=summary)

        return score

    return executive_summary()


# ---------------------------------------------------------------------------
# Tool sets
#
# Access policy:
#   plan      - NO filesystem tools; decomposes the question into research tasks.
#   research  - ONLY subagent permitted to read the dataset directory.
#   general   - Compute/write tools only (bash, text_editor); no dataset
#               readers. Synthesises from the plan + research output.
# ---------------------------------------------------------------------------

RESEARCH_TOOLS = [read_file(), list_files(), grep()]   # dataset access
PLAN_TOOLS     = []                                     # context-only
GENERAL_TOOLS  = [bash(), text_editor()]                # compute/write only


# ---------------------------------------------------------------------------
# Task 2 - Research subagent (standalone) — pipeline step 2
# ---------------------------------------------------------------------------

@task
def task_research_agent() -> Task:
    """Evaluate the research() subagent in isolation.

    Tests retrieval accuracy: given a structured research plan (sub-questions
    and retrieval directives from the plan subagent), does the research agent
    find ALL relevant documents from the corpus of ~127 JSON reports?
    """
    research_subagent = research(
        tools=RESEARCH_TOOLS,
        limits=[message_limit(RESEARCH_MESSAGE_LIMIT)],
        instructions=dedent(f"""\
            You are a document retrieval specialist with read-only access to
            a corpus of structured intelligence reports at:
            {DATASET_PATH}

            The corpus contains approximately 127 JSON files. Each file is a
            single intelligence report with these fields:
              serial       - unique 5-digit identifier (e.g. "00006")
              title        - report title
              topic        - category (cyber, military, missiles, weapons,
                             diplomacy, domestic, economy, space, tactics,
                             leadership, counterintelligence)
              author       - analyst who wrote the report
              classification - SECRET SQUIRREL or CONFIDENTIAL CHIPMUNK
              body         - full report text (most important for relevance)
              question     - the intelligence question this report answers
              tags         - keyword list

            You will receive a research plan containing numbered sub-questions
            and retrieval directives. Execute each directive against the corpus.

            RETRIEVAL STRATEGY — follow these steps for every plan sub-task:
              1. Call list_files("{DATASET_PATH}") once to get the complete file list.
              2. For EVERY JSON file in the listing, call read_file() to read it.
              3. For each sub-question in the plan, assess relevance by checking
                 'topic', 'title', 'question', and 'body' against that sub-question.
              4. Compile ALL relevant documents across all sub-questions.
              5. Do NOT stop reading early — check every file to ensure complete
                 retrieval. The corpus is ~127 files; read them all.
              6. Use grep() to follow up on specific names, events, or terms
                 identified in a sub-question if needed.

            For each relevant document found, report:
              - serial number (exact 5-digit string)
              - title
              - topic and classification
              - which plan sub-question(s) this document addresses
              - 2–3 sentence excerpt from 'body' directly relevant to that sub-question
              - author

            Do NOT fabricate document contents, serial numbers, or findings
            not present in the actual files.
            Do NOT report a document as relevant unless its body content
            actually addresses a sub-question in the plan.
            Never attempt to write or modify any files.
        """),
    )

    agent = deepagent(
        subagents=[research_subagent],
        memory=False,
        instructions=dedent(f"""\
            Delegate ALL retrieval work to the research subagent.
            Dataset: {DATASET_PATH}

            The input contains a research plan with specific sub-questions and
            retrieval directives. The research subagent must execute every
            directive and retrieve ALL documents relevant to each sub-question.

            Do NOT answer from prior knowledge.
            All findings must cite specific serial numbers from the actual files.
        """),
    )

    return Task(
        dataset=load_samples("research"),
        solver=as_solver(agent),
        scorer=[
            retrieval_accuracy_scorer(),
            task_completion_scorer(),
        ],
        model=AGENT_MODEL,
        metadata={"subagent": "research"},
        sandbox="local",
    )


# ---------------------------------------------------------------------------
# Task 1 - Plan subagent (standalone) — pipeline step 1
# ---------------------------------------------------------------------------

@task
def task_plan_agent() -> Task:
    """Evaluate the plan() subagent in isolation.

    The plan subagent has NO filesystem access. It receives the raw
    intelligence question and must decompose it into a structured set of
    targeted research sub-tasks that the research subagent will execute.
    Tests whether the planner produces a complete, actionable research plan.
    """
    plan_subagent = plan(
        tools=PLAN_TOOLS,
        instructions=dedent("""\
            You are an intelligence task decomposition specialist.

            You have NO access to any filesystem or dataset directory.
            Do NOT attempt to read, list, or search any files.

            Your task: given a raw intelligence question, decompose it into
            a structured research plan — a numbered list of specific, targeted
            sub-questions and retrieval directives that a document retrieval
            agent will execute against a corpus of intelligence reports.

            Each plan step must:
              - State one specific, answerable sub-question or retrieval goal.
              - Identify the topic domain or keywords to search for
                (e.g. "cyber", "military", "missiles", person names, events).
              - Be actionable: the retrieval agent must be able to execute it
                by scanning document fields (topic, title, question, body).

            Do NOT attempt to answer the question yourself.
            Do NOT invent document contents or serial numbers.
            Do NOT execute code or write any files.
            Output numbered steps only — no prose preamble.
        """),
    )

    agent = deepagent(
        subagents=[plan_subagent],
        memory=False,
        instructions=dedent("""\
            Delegate ALL task decomposition work to the plan subagent.
            The plan subagent receives the raw intelligence question and
            produces a structured research plan with specific sub-questions
            and retrieval directives. It has no filesystem access.
        """),
    )

    return Task(
        dataset=load_samples("plan"),
        solver=as_solver(agent),
        scorer=[
            task_completion_scorer(),
            model_graded_fact(model=JUDGE_MODEL),
        ],
        model=AGENT_MODEL,
        metadata={"subagent": "plan"},
        sandbox="local",
    )


# ---------------------------------------------------------------------------
# Task 3 - General subagent (standalone)
# ---------------------------------------------------------------------------

@task
def task_general_agent() -> Task:
    """Evaluate the general() subagent in isolation.

    The general subagent has NO dataset read access. Each input sample embeds
    a research summary AND an analysis plan so the subagent has all context
    it needs to write the final report. Tests faithfulness: every claim in the
    output must be attributed to a cited serial from the provided context.
    """
    general_subagent = general(
        tools=GENERAL_TOOLS,
        instructions=dedent("""\
            You are an intelligence report synthesis specialist.

            You have NO access to any dataset directory. Do NOT attempt
            to read_file, list_files, grep, or open any dataset path.
            Work strictly from the research summary and analysis plan
            provided in the task input.

            CRITICAL FAITHFULNESS RULES:
              1. Cite the source serial number for EVERY specific factual
                 claim using the format (serial: XXXXX).
              2. Do NOT add specific names, dates, figures, unit designations,
                 IP addresses, or technical details that are not present in
                 the provided research summary.
              3. If a claim cannot be attributed to a serial in the summary,
                 do not include it.
              4. Stay within the word limit stated in the task input.

            You may use bash to draft or format text.
            You may use text_editor to write the final report.
            Return the final report as structured markdown with an explicit
            Sources section listing every serial cited.
        """),
        memory="readwrite",
    )

    agent = deepagent(
        subagents=[general_subagent],
        memory=True,
        instructions=dedent("""\
            Delegate ALL synthesis work to the general subagent.
            The general subagent must work only from the research context
            and plan embedded in the task input — it has no filesystem access.

            Enforce faithfulness: ensure the final report cites a serial
            number for every specific factual claim and does not introduce
            details beyond the provided research summary.
        """),
    )

    return Task(
        dataset=load_samples("general"),
        solver=as_solver(agent),
        scorer=[
            faithfulness_scorer(),
            task_completion_scorer(),
        ],
        model=AGENT_MODEL,
        metadata={"subagent": "general"},
        sandbox="local",
    )


# ---------------------------------------------------------------------------
# Task 4 - Full DeepAgent (all three subagents collaborating)
# ---------------------------------------------------------------------------

@task
def task_deepagent_full() -> Task:
    """
    Evaluate the complete DeepAgent pipeline: plan -> research -> general.

    Three evaluation dimensions:
      1. Retrieval accuracy  — did research find all relevant documents?
      2. Faithfulness        — does the final report cite serials for every claim?
      3. End-to-end quality  — does the report correctly answer the question
                               (including correctly refusing hallucination traps)?
    """
    plan_subagent = plan(
        tools=PLAN_TOOLS,
        instructions=dedent("""\
            You are an intelligence task decomposition specialist.

            You have NO filesystem access. Do NOT attempt to read, list, or
            search any files.

            Given the raw intelligence question, decompose it into a structured
            research plan — a numbered list of specific, targeted sub-questions
            and retrieval directives for the research subagent to execute.

            Each plan step must:
              - State one specific, answerable sub-question or retrieval goal.
              - Identify relevant topic domains or keywords to search for
                (e.g. "cyber", "military", person names, unit names, events).
              - Be actionable: the retrieval agent executes it by scanning
                document fields (topic, title, question, body, tags).

            Do NOT attempt to answer the intelligence question yourself.
            Do NOT invent document contents or serial numbers.
            Output numbered steps only — no prose preamble.
        """),
    )

    research_subagent = research(
        tools=RESEARCH_TOOLS,
        limits=[message_limit(RESEARCH_MESSAGE_LIMIT)],
        instructions=dedent(f"""\
            You are a document retrieval specialist with read-only access to
            a corpus of ~127 JSON intelligence reports at:
            {DATASET_PATH}

            Each JSON file is a structured intelligence report with fields:
              serial        - unique 5-digit identifier (e.g. "00042")
              title         - report title
              topic         - category (cyber, military, missiles, weapons,
                              diplomacy, domestic, economy, space, tactics,
                              leadership, counterintelligence)
              body          - full report text (most important for relevance)
              question      - the intelligence question this report answers
              classification, author, tags

            You will receive a research plan with numbered sub-questions and
            retrieval directives. Execute every directive against the corpus.

            RETRIEVAL STRATEGY — mandatory for every plan sub-task:
              1. Call list_files("{DATASET_PATH}") once to get the complete list.
              2. Call read_file() on EVERY JSON file in the listing.
              3. For each sub-question, assess relevance via 'topic', 'title',
                 'question', and 'body' text.
              4. Compile ALL relevant documents across all sub-questions.
              5. Do not stop early — check every file.
              6. Use grep() to follow up on specific names, events, or terms
                 from a sub-question if needed.

            Your output must be a fully-cited research summary listing:
              - serial number, title, topic for each relevant document
              - which plan sub-question(s) each document addresses
              - A 2–3 sentence excerpt from 'body' directly relevant to that
                sub-question (so general never needs dataset access)

            CRITICAL: Do NOT fabricate serial numbers, titles, or body
            content. Do NOT report documents as relevant unless their actual
            body text addresses a sub-question in the plan.
        """),
    )

    general_subagent = general(
        tools=GENERAL_TOOLS,
        instructions=dedent("""\
            You are an intelligence report synthesis specialist.

            You have NO access to the dataset directory. Do NOT attempt to
            read_file, list_files, grep, or open any dataset path.
            Work only from the research summary and plan passed to you.

            CRITICAL FAITHFULNESS RULES:
              1. Every specific factual claim MUST include a serial citation
                 in the format (serial: XXXXX).
              2. Do NOT add names, dates, figures, unit designations, or
                 technical details not present in the research summary.
              3. If a claim cannot be attributed to a serial, omit it.
              4. For questions about entities or events not found in the
                 research summary, explicitly state they were not found
                 rather than fabricating an answer.

            Use text_editor to write the final markdown report.
            End the report with an explicit "## Sources" section listing
            every serial cited, its title, and a one-line description.
        """),
        memory="readwrite",
    )

    agent = deepagent(
        subagents=[plan_subagent, research_subagent, general_subagent],
        memory=True,
        todo_write=True,
        max_depth=4,
        instructions=dedent(f"""\
            You orchestrate three specialised subagents with strict access
            boundaries to produce a grounded intelligence assessment.

            Pipeline order: plan → research → general

              plan     : No filesystem access. Receives the raw intelligence
                         question and decomposes it into a structured list of
                         targeted research sub-questions and retrieval directives.

              research : ONLY subagent that may read the dataset at
                         {DATASET_PATH}
                         Receives the plan's sub-questions and retrieves ALL
                         relevant documents for each directive. Produces a
                         fully-cited research summary.

              general  : No filesystem access. Receives the plan and the
                         research summary. Writes the final report. MUST cite
                         serial numbers for every factual claim. MUST explicitly
                         refuse to fabricate details for entities or events not
                         found in the research summary.

            Mandatory workflow:
              1. Invoke plan with the user's raw intelligence question.
                 Collect the numbered list of research sub-questions.
              2. Pass the plan to research. Verify the output contains serial
                 numbers and body excerpts covering every sub-question.
              3. Pass both the plan and research output to general. Collect
                 the final report.
              4. Verify the final report has serial citations for all major
                 claims. If any section lacks citations, return to general
                 to add them.
              5. Return the final report as the task output.
        """),
    )

    return Task(
        dataset=load_samples("deepagent"),
        solver=as_solver(agent),
        scorer=[
            retrieval_accuracy_scorer(),
            faithfulness_scorer(),
            task_completion_scorer(),
            executive_summary_scorer(),
        ],
        model=AGENT_MODEL,
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