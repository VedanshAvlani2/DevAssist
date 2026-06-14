from crewai import Agent

def make_coder_agent():
    return Agent(
        role="Software Engineer",
        goal=(
            "Write clean, working Python code that implements the requested changes. "
            "Use the retrieved code context to stay consistent with the existing codebase style. "
            "Output ONLY the complete updated file contents, no explanations, no markdown fences."
        ),
        backstory=(
            "You are a senior Python engineer. "
            "You receive a task, subtasks, and relevant code snippets from the codebase. "
            "You write precise, production-ready code that fits seamlessly into the existing project."
        ),
        verbose=True,
        allow_delegation=False,
    )
