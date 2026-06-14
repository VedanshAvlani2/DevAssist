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

# Read from env — override via: GITHUB_REPO=user/repo python -m rag.indexer
GITHUB_REPO = os.environ.get("GITHUB_REPO", "VedanshAvlani2/DevAssist")
REPO_NAME = GITHUB_REPO.replace("/", "_")
CLONE_DIR = f"./indexed_repos/{REPO_NAME}"

# Clone repo if not already cloned
if not os.path.exists(CLONE_DIR):
    print(f"Cloning {GITHUB_REPO} into {CLONE_DIR}...")
    subprocess.run(
        ["git", "clone", f"https://github.com/{GITHUB_REPO}.git", CLONE_DIR],
        check=True
    )

REPO_PATH = CLONE_DIR

ALLOWED_EXTENSIONS = {".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml"}
SKIP_DIRS = {"venv", ".git", "__pycache__", "node_modules", ".env"}


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


def index_repo():
    print(f"\nIndexing repo: {GITHUB_REPO}")
    print(f"Collection name: {REPO_NAME}\n")

    documents = load_files_from_repo(REPO_PATH)
    print(f"\nTotal files loaded: {len(documents)}")

    if not documents:
        print("No files found. Check REPO_PATH or clone failed.")
        return

    chunks = chunk_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Delete only THIS repo's collection — other repos untouched
    try:
        chroma_client.delete_collection(REPO_NAME)
        print(f"Deleted old index for {REPO_NAME}.")
    except Exception:
        pass

    collection = chroma_client.create_collection(REPO_NAME)

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

    print(f"\nDone. {len(chunks)} chunks indexed as '{REPO_NAME}'.")


if __name__ == "__main__":
    index_repo()