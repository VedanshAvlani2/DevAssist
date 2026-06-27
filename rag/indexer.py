import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
import subprocess
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ALLOWED_EXTENSIONS = {".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml"}
SKIP_DIRS = {"venv", "venv312", ".git", "__pycache__", "node_modules", ".env"}


def load_files_from_repo(repo_path: str) -> list[dict]:
    documents = []
    repo = Path(repo_path)

    for file_path in repo.rglob("*"):
        if file_path.is_dir():
            continue
        if any(skip in file_path.parts for skip in SKIP_DIRS):
            continue
        if file_path.suffix not in ALLOWED_EXTENSIONS:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                documents.append({
                    "path": str(file_path),
                    "content": content
                })
                print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Skipped {file_path}: {e}")

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, split in enumerate(splits):
            chunks.append({
                "id": f"{doc['path']}::chunk{i}",
                "content": split,
                "source": doc["path"]
            })

    return chunks


def index_repo(github_repo: str = None):
    """
    Clone (if needed) and index a GitHub repository into ChromaDB.

    Args:
        github_repo: Repo in 'owner/repo' format. Falls back to GITHUB_REPO env var.
    """
    if github_repo is None:
        github_repo = os.environ.get("GITHUB_REPO", "VedanshAvlani2/DevAssist")

    repo_name = github_repo.replace("/", "_")
    clone_dir = f"./indexed_repos/{repo_name}"

    if not os.path.exists(clone_dir):
        print(f"Cloning {github_repo} into {clone_dir}...")
        subprocess.run(
            ["git", "clone", f"https://github.com/{github_repo}.git", clone_dir],
            check=True
        )

    print(f"\nIndexing repo: {github_repo}")
    print(f"Collection name: {repo_name}\n")

    documents = load_files_from_repo(clone_dir)
    print(f"\nTotal files loaded: {len(documents)}")

    if not documents:
        print("No files found. Check the clone directory or repo contents.")
        return

    chunks = chunk_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    chroma_client = chromadb.HttpClient(host="localhost", port=8000)

    try:
        chroma_client.delete_collection(repo_name)
        print(f"Deleted old index for {repo_name}.")
    except Exception:
        pass

    collection = chroma_client.create_collection(repo_name)

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        texts = [c["content"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [{"source": c["source"]} for c in batch]

        vectors = embeddings_model.embed_documents(texts)

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas
        )

        print(f"Indexed chunks {i} to {i + len(batch)}")

    print(f"\nDone. {len(chunks)} chunks indexed as '{repo_name}'.")


if __name__ == "__main__":
    index_repo()
