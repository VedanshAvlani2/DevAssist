# agents/crew.py
import os
from dotenv import load_dotenv
from crewai import Crew, Task, Process

from agents.planner import make_planner_agent
from agents.retriever import make_retriever_agent
from agents.coder import make_coder_agent
from agents.reviewer import make_reviewer_agent
from agents.pr_agent import make_pr_agent

load_dotenv()


def run_devassist_crew(task_description: str) -> str:
    """
    Entry point. Runs the full CrewAI pipeline for a given task.
    Returns the final result (PR URL or error).
    """

    # ── Agents ──────────────────────────────────────────────────────────────
    planner   = make_planner_agent()
    retriever = make_retriever_agent()
    coder     = make_coder_agent()
    reviewer  = make_reviewer_agent()
    pr_agent  = make_pr_agent()

    # ── Tasks ────────────────────────────────────────────────────────────────
    plan_task = Task(
        description=(
            f"Break this coding task into subtasks and identify the target file.\n\n"
            f"Task: {task_description}\n\n"
            f"Return JSON with keys: subtasks (list of strings), target_file (string)."
        ),
        expected_output="JSON object with subtasks list and target_file string.",
        agent=planner,
    )

    retrieve_task = Task(
        description=(
            f"Search the codebase for code relevant to this task:\n\n"
            f"Task: {task_description}\n\n"
            f"Use the RAG Retriever Tool. Return the most relevant code chunks."
        ),
        expected_output="Relevant code chunks from the indexed codebase.",
        agent=retriever,
    )

    code_task = Task(
        description=(
            f"Write the code changes for this task:\n\n"
            f"Task: {task_description}\n\n"
            f"Use the plan from the Planner and code context from the Retriever. "
            f"Output ONLY the complete updated file contents."
        ),
        expected_output="Complete updated Python file contents, no markdown, no explanation.",
        agent=coder,
        context=[plan_task, retrieve_task],
    )

    review_task = Task(
        description=(
            f"Review the generated code for correctness and completeness.\n\n"
            f"Original task: {task_description}\n\n"
            f"Output APPROVED or REVISION NEEDED: <reason>."
        ),
        expected_output="APPROVED or REVISION NEEDED: <reason>",
        agent=reviewer,
        context=[plan_task, code_task],
    )

    pr_task = Task(
        description=(
            f"If the review output starts with APPROVED, create a GitHub PR.\n\n"
            f"Use the GitHub PR Tool with this exact format:\n"
            f"target_file|generated_code|task_description\n\n"
            f"Get target_file from the Planner output.\n"
            f"Get generated_code from the Coder output.\n"
            f"task_description = '{task_description}'\n\n"
            f"If review says REVISION NEEDED, return the reviewer's reason instead."
        ),
        expected_output="GitHub PR URL or revision reason.",
        agent=pr_agent,
        context=[plan_task, code_task, review_task],
    )

    # ── Crew ─────────────────────────────────────────────────────────────────
    crew = Crew(
        agents=[planner, retriever, coder, reviewer, pr_agent],
        tasks=[plan_task, retrieve_task, code_task, review_task, pr_task],
        process=Process.sequential,   # tasks run in order
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    task = input("Enter task: ")
    output = run_devassist_crew(task)
    print("\n── Final Result ──")
    print(output)