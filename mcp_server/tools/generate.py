# mcp_server/tools/generate.py

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


def generate_code(instruction: str, context: str = "") -> str:
    """
    Generates Python code for the given instruction using GPT-4o.
    Optionally accepts retrieved code context for better results.
    All logging goes to stderr to keep MCP stdout clean.
    """
    sys.stderr.write(f"[generate] Generating code for: {instruction[:80]}...\n")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=os.environ["OPENAI_API_KEY"]
    )

    system_prompt = """You are an expert Python developer.
Given a task and relevant code context, produce ONLY the complete updated Python file.
No explanations. No markdown fences. No preamble. Raw Python code only."""

    user_prompt = f"""Task: {instruction}

Relevant code context:
{context}

Output the complete updated Python file:"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    generated = response.content.strip()

    # Strip markdown fences if model wraps output
    if generated.startswith("```"):
        lines = generated.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        generated = "\n".join(lines).strip()

    sys.stderr.write(f"[generate] Generated {len(generated)} chars\n")
    return generated
