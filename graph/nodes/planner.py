# graph/nodes/planner.py
# This node receives the raw task string from the user.
# It uses GPT-4o to:
#   1. Break the task into clear subtasks
#   2. Identify which file should be modified
# It returns an updated state dict.

import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import DevAssistState


def plan_task(state: DevAssistState) -> DevAssistState:
    """
    Node 1: Plan
    Input:  state["task"] — e.g. "Add input validation to the login function"
    Output: state["subtasks"], state["target_file"]
    """

    task = state["task"]

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"]
    )

    system_prompt = """You are a senior software engineer planning code changes.
Given a task description, respond ONLY with a JSON object with two keys:
- "subtasks": a list of 2-5 concrete implementation steps (strings)
- "target_file": the most likely Python file path to modify (e.g. "src/auth.py"), or null if unknown

Do not include any explanation. Only output valid JSON."""

    user_prompt = f"Task: {task}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    try:
        # Strip markdown code fences if GPT-4o wraps response in ```json ... ```
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]  # get content between first pair of backticks
            if raw.startswith("json"):
                raw = raw[4:]          # strip the word "json"
        plan = json.loads(raw.strip())
        subtasks = plan.get("subtasks", [])
        target_file = plan.get("target_file", None)
    except json.JSONDecodeError:
        subtasks = [response.content]
        target_file = None

    print(f"[planner] Subtasks: {subtasks}")
    print(f"[planner] Target file: {target_file}")

    return {
        **state,
        "subtasks": subtasks,
        "target_file": target_file
    }