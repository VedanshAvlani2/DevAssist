from crewai import Agent

def make_reviewer_agent():
    return Agent(
        role="Code Reviewer",
        goal=(
            "Review generated Python code for correctness, style, and completeness. "
            "Check: does it implement the task? Are there syntax errors? "
            "Does it match the codebase style from the retrieved context? "
            "Output: APPROVED if code is acceptable, or REVISION NEEDED: <reason> if not."
        ),
        backstory=(
            "You are a meticulous code reviewer with 10 years of Python experience. "
            "You catch bugs, style violations, and incomplete implementations. "
            "You give concise, actionable feedback."
        ),
        verbose=True,
        allow_delegation=False,
    )
