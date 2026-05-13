# DeepAgent Evaluation with Inspect AI

A generic deep-research agent evaluation harness built on
[inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai).
It exercises the full `deepagent()` pipeline as well as each of its three
subagents individually against any corpus you point at — the questions live in
a JSON samples file, so swapping domain is a question of swapping samples.

This README documents the two files in this directory that drive the harness:

- `deep_agent_inspect.py` — solvers, scorers, and `@task` definitions
- `deepagent_samples.json` — the evaluation dataset (questions + grading metadata)

For the LangGraph reference implementation see `deep_research.py`; for the
single-pipeline Inspect variant see `deep_research_inspect.py`.

---

## Quickstart

```bash
# 1. Set the LAS proxy token (the harness maps OPENAI_KEY -> OPENAI_API_KEY).
export OPENAI_KEY=<LAS_API_token>

# 2. (Optional) point at a different corpus / judge model.
export DEEP_RESEARCH_DATASET="/path/to/your/corpus"
export DEEP_RESEARCH_JUDGE_MODEL="openai/gpt-5-mini"

# 3. Activate the venv that has inspect-ai installed.
source .venv/bin/activate

# 4. Run all four tasks.
inspect eval deep_agent_inspect.py --model openai/gpt-5-mini

# Or run one task at a time:
inspect eval deep_agent_inspect.py@task_research_agent  --model openai/gpt-5-mini
inspect eval deep_agent_inspect.py@task_plan_agent      --model openai/gpt-5-mini
inspect eval deep_agent_inspect.py@task_general_agent   --model openai/gpt-5-mini
inspect eval deep_agent_inspect.py@task_deepagent_full  --model openai/gpt-5-mini

# View results afterwards.
inspect view
```

## Environment

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_KEY` | _required_ | LAS proxy token. Auto-mapped to `OPENAI_API_KEY` if the latter is unset. |
| `OPENAI_API_KEY` | — | Standard OpenAI-compatible API key (overrides `OPENAI_KEY`). |
| `OPENAI_BASE_URL` | `https://llm-west.ncsu-las.net/v1` | API endpoint; defaults to the LAS proxy. |
| `DEEP_RESEARCH_DATASET` | the Zendia Clean Datasets path | Directory the agent reads from. |
| `DEEP_RESEARCH_JUDGE_MODEL` | `openai/gpt-5-mini` | Model used by every LLM-judge scorer. |

---

## `deep_agent_inspect.py`

### Tasks

Four `@task` functions, each evaluating a different slice of the pipeline:

| Task | Subagents wired | Samples drawn (by role) | Scorers |
| --- | --- | --- | --- |
| `task_research_agent` | `research` only (read-only tools) | `research` | task_completion, research_rubric, research_faithfulness, research_citations, topic_relevance, required_citations |
| `task_plan_agent` | `plan` only (read-only tools) | `plan` | task_completion, plan_rubric, plan_relevance |
| `task_general_agent` | `general` only (read/write tools) | `general` | task_completion, general_rubric |
| `task_deepagent_full` | `research` → `plan` → `general` orchestrated by `deepagent()` | `deepagent` | task_completion, deepagent_rubric, research_faithfulness, research_citations, topic_relevance, required_citations |

Each task runs in the `"local"` sandbox and uses `as_solver(deepagent(...))` so
all standard Inspect tooling (transcripts, `inspect view`, log replay) applies.

### Tool sets

```python
READONLY_TOOLS  = [read_file(), list_files(), grep()]
READWRITE_TOOLS = [read_file(), list_files(), grep(), bash(), text_editor()]
```

The `research` and `plan` subagents are restricted to `READONLY_TOOLS`; the
`general` subagent gets `READWRITE_TOOLS` so it can use `bash` for aggregation
and `text_editor` to scratch files.

### Scorers

Every scorer is registered with `@scorer(metrics=[mean(), stderr()])` so
inspect reports both a mean and standard error across the dataset.

| Scorer | Kind | What it measures |
| --- | --- | --- |
| `task_completion_scorer` | LLM judge (C/P/I → 1.0/0.5/0.0) | Coarse "did the response satisfy the target." Backwards-compatible signal. |
| `score_research_rubric` | LLM judge (1–5, normalized) | Research role: faithfulness, citations, coverage, honest gaps, read-only discipline. |
| `score_plan_rubric` | LLM judge (1–5) | Plan role: structure, specificity, atomicity, success criteria, no-execution. |
| `score_general_rubric` | LLM judge (1–5) | General role: correctness, format compliance, faithfulness, citations. |
| `score_deepagent_rubric` | LLM judge (1–5) | Full pipeline: research→plan→execute evidence, structure, sources list. |
| `score_plan_relevance` / `score_topic_relevance` | LLM judge (1–5) | Sample-specific topic coverage. Reads `expected_topics` and `forbidden_topics` from sample metadata. `score_topic_relevance` is an alias for use on non-plan tasks. |
| `score_required_citations` | Deterministic (fraction in [0,1]) | Reference preservation. Reads `required_citations` from metadata and checks each token appears in the response. |
| `score_research_faithfulness` | LLM judge (1–5) | Tool-trace faithfulness. Reconstructs everything the agent fetched via `read_file` / `grep` from `state.messages`, judges the summary against that concatenated evidence. |
| `score_research_citations` | LLM judge per citation, mean (1–5) | Per-citation verification. Parses citation tokens out of the response, resolves each one to a file on disk under `DEEP_RESEARCH_DATASET`, judges every citation independently. |

All scorers tolerate missing metadata gracefully: those that depend on
`expected_topics` / `required_citations` return `Score(value=0.0, metadata={"skipped": True})`
with an explanation if the sample doesn't carry the required fields, so adding
new test types is purely additive.

### Citation parsing

`_extract_citations` is tuned for the current Zendia naming convention:

- `file:///abs/path.json` URIs
- bare `*.json` paths (resolved relative to `DEEP_RESEARCH_DATASET`)
- 5-digit serials (`00152`) — resolved by globbing the dataset root for `{serial}.json`

If you switch corpora, edit `_CITE_RE` and `_resolve_citation` to match the new
identifier format. Everything else in the file is corpus-agnostic.

---

## `deepagent_samples.json`

A list of evaluation samples, one JSON object per sample. The file is read by
`load_samples(role)` which filters by the top-level `role` field — each task
pulls only the samples for its role.

### Required fields

```json
{
  "id": "research_04",
  "role": "research",
  "input": "Full prompt the agent sees.",
  "target": "Reference description of what a good response covers."
}
```

| Field | Type | Purpose |
| --- | --- | --- |
| `id` | string | Stable identifier shown in the eval log. |
| `role` | `"research" \| "plan" \| "general" \| "deepagent"` | Which task picks this sample up. |
| `input` | string | The prompt. Must be self-contained (corpus path, instructions, format). |
| `target` | string | Plain-language description of a good answer. Used by `task_completion_scorer` and shown to every LLM judge. |

### Optional `metadata` fields

`metadata` is free-form, but specific keys are consumed by specific scorers:

| Key | Consumed by | Effect |
| --- | --- | --- |
| `expected_topics: list[str]` | `score_plan_relevance` / `score_topic_relevance` | Each item is a "must-cover" theme; the judge grades yes / partial / no per item. Missing → scorer skipped. |
| `forbidden_topics: list[str]` | same | Decoys the response should NOT drift into. Each hit deducts 0.2 from the coverage fraction. Optional even when `expected_topics` is set. |
| `required_citations: list[str]` | `score_required_citations` | Tokens that must appear literally in the response (case-insensitive). Score = `hits / required`. Missing → scorer skipped. |
| `test_kind: str` | informational only | Tag for filtering / analysis (`summarization_relevance`, `reference_preservation`, `plan_topic_relevance`). |
| `topic`, `dataset_path`, `difficulty`, ... | informational only | Not consumed by any scorer; useful for stratification when reading logs. |

### Current sample inventory

| ID | Role | test_kind | expected | forbidden | required_cites |
| --- | --- | --- | --- | --- | --- |
| `research_01` | research | – | 4 | 3 | – |
| `research_02` | research | – | 6 | 2 | `00152`, `00116` |
| `research_03` | research | – | 5 | 2 | – |
| `research_04` | research | **summarization_relevance** | 4 | 4 | – |
| `research_05` | research | **reference_preservation** | 4 | 2 | `00152` |
| `plan_01` | plan | – | 7 | 3 | – |
| `plan_02` | plan | – | 7 | 3 | – |
| `plan_03` | plan | **plan_topic_relevance** | 6 | 4 | – |
| `general_01` | general | – | – | – | – |
| `general_02` | general | – | – | – | – |
| `deep_01` | deepagent | – | 6 | 2 | – |
| `deep_02` | deepagent | – | 5 | 2 | `00152`, `00116` |

The three test_kind-tagged samples (`research_04`, `research_05`, `plan_03`)
are the dedicated regression tests for summarization relevance, reference
preservation, and plan-topic relevance respectively. Every other sample also
contributes to the same scorers via its `expected_topics` / `required_citations`
metadata, so the signal accumulates across the full run.

---

## Extending the harness

**Add a sample.** Drop a new object into `deepagent_samples.json` with the
required four fields plus whatever metadata you want graded. No code change
needed.

**Add a check.**
1. Pick a metadata key (e.g. `must_use_tags`) and an interpretation.
2. Add a `@scorer(metrics=[mean(), stderr()])` function that reads the key off
   `state.metadata` and returns a `Score`. Return `Score(value=0.0, metadata={"skipped": True})`
   if the key is absent so older samples still pass cleanly.
3. Append the scorer to the `scorer=[...]` list of whichever `@task` should run it.

**Swap corpora.** Set `DEEP_RESEARCH_DATASET` to the new directory and replace
the samples in `deepagent_samples.json`. If the new corpus uses different
citation identifiers (UUIDs, DOIs, etc.), update `_CITE_RE` and
`_resolve_citation` in `deep_agent_inspect.py`.

**Swap judge model.** `DEEP_RESEARCH_JUDGE_MODEL` overrides the default for
every LLM judge. The model needs to follow JSON-only instructions reliably
(`gpt-5-mini`, `claude-haiku-4-5`, etc.) — `_parse_json` is forgiving but
non-JSON judges score the response 0.

---

## Scoring summary at a glance

For a research-task sample with all metadata set, a single run produces six
independent signals:

1. **task_completion** — coarse pass / partial / fail
2. **research_rubric** — judge's overall quality grade
3. **research_faithfulness** — claims vs. fetched evidence (tool trace)
4. **research_citations** — per-citation verification against on-disk files
5. **topic_relevance** — coverage of `expected_topics`, penalized by `forbidden_topics`
6. **required_citations** — deterministic presence check for `required_citations`

A summary that's coherent, well-cited, on-topic, and grounded in what the
agent actually read scores high on all six. A summary that's confidently wrong
but well-cited scores high on (4) and (6) and low on (3) — exactly the kind of
failure citation-only checks miss.
