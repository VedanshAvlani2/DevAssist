import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_relevant_context(query: str, n_results: int = 5) -> str:
    """
    Search the indexed codebase for chunks relevant to the given query.

    Reads GITHUB_REPO from the environment at call time so repo switches
    between requests are reflected immediately.

    Args:
        query: Natural language description of what code to find.
        n_results: Number of chunks to return.

    Returns:
        Formatted string of matching code chunks, separated by '---'.
    """
    github_repo = os.environ.get("GITHUB_REPO", "VedanshAvlani2/DevAssist")
    repo_name = github_repo.replace("/", "_")

    client = chromadb.HttpClient(host="localhost", port=8000)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )

    collection = client.get_collection(
        name=repo_name,
        embedding_function=openai_ef
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    chunks = results["documents"][0]
    formatted = "\n---\n".join(chunks)
    return formatted
