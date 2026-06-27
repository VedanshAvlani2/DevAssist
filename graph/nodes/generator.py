# graph/nodes/generator.py
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import DevAssistState


def _read_target_file(target_file: str) -> str:
    """
    Try to read the target file from the cloned repo so the generator
    works from the real source code rather than hallucinating.
    Falls back to empty string if not found.
    """
    if not target_file:
        return ""

    github_repo = os.environ.get("GITHUB_REPO", "")
    if github_repo:
        repo_name = github_repo.replace("/", "_")
        clone_path = Path(f"./indexed_repos/{repo_name}") / target_file
        if clone_path.exists():
            return clone_path.read_text(encoding="utf-8", errors="ignore")

    # Fallback: check local working directory
    local_path = Path(target_file)
    if local_path.exists():
        return local_path.read_text(encoding="utf-8", errors="ignore")

    return ""


def generate_code_node(state: DevAssistState) -> DevAssistState:
    task = state["task"]
    context = state.get("retrieved_context", "")
    target_file = state.get("target_file") or ""

    print(f"[generator] Generating code for: {task[:80]}...")

    # Read the actual current file so the model edits it, not hallucinate it
    original_source = _read_target_file(target_file)
    if original_source:
        print(f"[generator] Loaded original source for {target_file} ({len(original_source)} chars)")
    else:
        print(f"[generator] No original source found for '{target_file}' — generating from scratch")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=os.environ["OPENAI_API_KEY"]
    )

    system_prompt = """You are an expert Python developer making precise edits to existing code.
You will be given:
  - A task description
  - The CURRENT file contents (source of truth — preserve everything not explicitly changed)
  - Relevant codebase context for reference

Rules:
  - Output ONLY the complete updated file, preserving all existing logic unless the task says to change it
  - No explanations, no markdown fences, no preamble
  - Raw Python code only"""

    original_block = (
        f"Current file contents of `{target_file}`:\n```python\n{original_source}\n```"
        if original_source
        else f"No existing file found for `{target_file}` — create it from scratch."
    )

    user_prompt = f"""Task: {task}

{original_block}

Relevant codebase context (for reference):
{context}

Output the complete updated file:"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    generated = response.content.strip()

    if generated.startswith("```"):
        lines = generated.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        generated = "\n".join(lines).strip()

    print(f"[generator] Generated {len(generated)} chars")

    return {
        **state,
        "generated_code": generated
    }
