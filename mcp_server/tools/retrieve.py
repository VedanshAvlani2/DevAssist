# mcp/tools/retrieve.py

import sys
import os

# Add project root to path so we can import rag/retriever.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from rag.retriever import get_relevant_context


def retrieve_code(query: str, n_results: int = 5) -> str:
    """
    Takes a natural language query.
    Returns top matching code chunks from ChromaDB as a formatted string.
    """
    result = get_relevant_context(query, n_results=n_results)

    if not result or not result.strip():
        return "No relevant code found in the repository index."

    return result