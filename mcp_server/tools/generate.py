# graph/nodes/generator.py
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import DevAssistState

def generate_code(state: DevAssistState) -> DevAssistState:
    task = state["task"]
    context = state.get("retrieved_context", "")

    print(f"[generator] Generating code for: {task[:80]}...")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=os.environ["OPENAI_API_KEY"]
    )

    system_prompt = """You are an expert Python developer.
Given a task and relevant code context, produce ONLY the complete updated Python file.
No explanations. No markdown fences. No preamble. Raw Python code only."""

    user_prompt = f"""Task: {task}

Relevant code context:
{context}

Output the complete updated Python file:"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    generated = response.content.strip()

    # Strip markdown fences if GPT-4o wraps output
    if generated.startswith("```"):
        lines = generated.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        generated = "\n".join(lines).strip()

    print(f"[generator] Generated {len(generated)} chars")

    return {
        **state,
        "generated_code": generated
    }