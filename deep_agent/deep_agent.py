from textwrap import dedent
from inspect_ai import Task, task
from inspect_ai.agent import deepagent
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import includes
from inspect_ai.tool import bash, text_editor

@task
def deep_agent():
    return Task(
        dataset=json_dataset("zendia_questions.json"),
        solver=deepagent(
            tools=[bash(), text_editor()]
        ),
        scorer=includes(),
        sandbox="local",
    )