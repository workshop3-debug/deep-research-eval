# DeepAgent Evaluation with Inspect AI

A deep-research agent evaluation harness built on
[inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai).
It exercises the full `deepagent()` pipeline as well as each of its three
subagents individually against a corpus of structured intelligence reports.
Questions and grading targets live in `deepagent_samples.json`, so swapping
domains is a matter of swapping samples.

This README documents the two files in this directory that drive the harness:

- `deep_agent_inspect.py` — solvers, scorers, and `@task` definitions
- `deepagent_samples.json` — the evaluation dataset (questions + grading targets)

---

## Quickstart

```bash
# 1. Activate the venv that has inspect-ai installed.
source .venv/bin/activate

# 2. Run all four tasks.
inspect eval deep_agent_inspect.py --model openai/deepinfra/anthropic/claude-4-sonnet

# Or run one task at a time:
inspect eval deep_agent_inspect.py@task_research_agent  --model openai/deepinfra/anthropic/claude-4-sonnet
inspect eval deep_agent_inspect.py@task_plan_agent      --model openai/deepinfra/anthropic/claude-4-sonnet
inspect eval deep_agent_inspect.py@task_general_agent   --model openai/deepinfra/anthropic/claude-4-sonnet
inspect eval deep_agent_inspect.py@task_deepagent_full  --model openai/deepinfra/anthropic/claude-4-sonnet

# View results afterwards.
inspect view
```

## Configuration

The following constants are set at the top of `deep_agent_inspect.py`:

| Constant | Default | Purpose |
| --- | --- | --- |
| `DATASET_PATH` | `/home/workshop11/deep_agent_eval/1_5_3_GPT 4` | Corpus directory the research agent reads from. |
| `AGENT_MODEL` | `openai/deepinfra/anthropic/claude-4-sonnet` | Model passed to each `deepagent()` / subagent solver. |
| `JUDGE_MODEL` | `openai/gpt-4o-mini` | Model used by every LLM-judge scorer. |
| `RESEARCH_MESSAGE_LIMIT` | `300` | Max model turns for the research subagent per invocation. |

---

## `deep_agent_inspect.py`

### Pipeline order

```
plan → research → general
```

- **plan** — decomposes the raw question into a numbered list of retrieval directives (no filesystem access).
- **research** — reads the corpus and produces a fully-cited research summary.
- **general** — synthesises the plan + research summary into a final report (no corpus access; uses `bash` / `text_editor`).

### Tool sets

```python
PLAN_TOOLS     = []                                  # no filesystem access
RESEARCH_TOOLS = [read_file(), list_files(), grep()] # dataset access only
GENERAL_TOOLS  = [bash(), text_editor()]             # compute/write, no dataset reads
```

### Tasks

Four `@task` functions, each evaluating a different slice of the pipeline:

| Task | Subagents wired | Samples drawn (by role) | Scorers |
| --- | --- | --- | --- |
| `task_research_agent` | `research` only | `research` | `retrieval_accuracy`, `task_completion` |
| `task_plan_agent` | `plan` only | `plan` | `task_completion`, `model_graded_fact` |
| `task_general_agent` | `general` only | `general` | `faithfulness`, `task_completion` |
| `task_deepagent_full` | `plan` → `research` → `general` | `deepagent` | `retrieval_accuracy`, `faithfulness`, `task_completion`, `executive_summary` |

Each task runs in the `"local"` sandbox and uses `as_solver(deepagent(...))`.

### Scorers

Every scorer is registered with `@scorer(metrics=[accuracy()])`.

| Scorer | Kind | What it measures |
| --- | --- | --- |
| `task_completion_scorer` | LLM judge (C/P/I → 1.0/0.5/0.0) | Did the response fully satisfy the target? |
| `retrieval_accuracy_scorer` | LLM judge (C/P/I → 1.0/0.5/0.0) | Did the research agent cite all expected serial numbers with key findings? |
| `faithfulness_scorer` | LLM judge (C/P/I → 1.0/0.5/0.0) | Are all specific factual claims attributed to cited serial numbers? Flags fabrication and hallucination-trap failures. |
| `executive_summary_scorer` | Non-scoring (always 1.0) | Calls the judge model to write a 3–5 sentence executive summary of the agent's output. Stored in `Score.explanation`; visible in `inspect view`. Does not affect numeric metrics. |

All three judge scorers use the same C/P/I rubric:

- **C** (correct / complete) → 1.0
- **P** (partial) → 0.5
- **I** (incorrect / missing) → 0.0

---

## `deepagent_samples.json`

A list of evaluation samples, one JSON object per sample. The file is read by
`load_samples(role)`, which filters by the top-level `role` field.

The string `{DATASET_PATH}` anywhere in `input`, `target`, or `metadata` values
is substituted at load time with the `DATASET_PATH` constant, keeping the JSON
file path-agnostic.

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
| `input` | string | The prompt sent to the agent. Must be self-contained. |
| `target` | string | Plain-language description of a good answer. Shown to every LLM judge. |

### Optional `metadata` fields

`metadata` is free-form. No scorer currently reads any metadata key; fields
like `expected_topics`, `required_citations`, and `test_kind` are informational
and useful for log stratification but do not affect scoring.

---

## Extending the harness

**Add a sample.** Drop a new object into `deepagent_samples.json` with the
required four fields. No code change needed.

**Add a scorer.**
1. Write a `@scorer(metrics=[accuracy()])` function that inspects `state` and
   returns a `Score`.
2. Append it to the `scorer=[...]` list of whichever `@task` should run it.

**Change the corpus.** Update `DATASET_PATH` and replace the samples in
`deepagent_samples.json`. Update the research subagent's instructions if the
new corpus uses a different JSON schema.

**Change the judge model.** Edit `JUDGE_MODEL` at the top of the file.
The model must reliably output `GRADE: C`, `GRADE: P`, or `GRADE: I`.

---

## Scoring summary at a glance

For a `deepagent` sample a single run produces four signals:

1. **retrieval_accuracy** — did research find all relevant documents?
2. **faithfulness** — does the final report cite serials for every claim?
3. **task_completion** — coarse pass / partial / fail against the target
4. **executive_summary** — non-scoring; a plain-language summary of what the
   agent found, stored in `Score.explanation` for quick human review in
   `inspect view`.
