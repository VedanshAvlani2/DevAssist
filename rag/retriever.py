import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "VedanshAvlani2/DevAssist")
REPO_NAME = GITHUB_REPO.replace("/", "_")

def get_relevant_context(query: str, n_results: int = 5) -> str:
    client = chromadb.HttpClient(host="localhost", port=8000)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )

    collection = client.get_collection(
        name=REPO_NAME,
        embedding_function=openai_ef
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    chunks = results["documents"][0]
    formatted = "\n---\n".join(chunks)
    return formatted