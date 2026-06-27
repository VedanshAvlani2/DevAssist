"""
DevAssist Web — FastAPI backend
================================
Run with:
    venv312\\Scripts\\activate
    uvicorn web.app:app --host 0.0.0.0 --port 8080 --reload

Endpoints
---------
POST  /repo/setup          — Start indexing a GitHub repo in the background
GET   /repo/status         — Poll indexing progress
POST  /chat                — Send a task; returns diff + message_id
POST  /chat/approve/{id}   — Approve a pending diff and push a PR
GET   /chat/history        — Full conversation history
DELETE /chat/history       — Clear conversation history
GET   /                    — Serve the chat UI
"""

import os
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="DevAssist")

# ── State (in-memory, backed by JSON file) ───────────────────────────────────

_indexing_status: dict = {"status": "idle", "repo": None, "message": "No repo indexed yet."}
_chat_history: list = []
_pending: dict = {}  # message_id → {target_file, generated_code, task}

HISTORY_FILE = Path(__file__).parent / "chat_history.json"


def _load_history():
    global _chat_history
    if HISTORY_FILE.exists():
        try:
            _chat_history = json.loads(HISTORY_FILE.read_text())
        except Exception:
            _chat_history = []


def _save_history():
    HISTORY_FILE.write_text(json.dumps(_chat_history, indent=2))


_load_history()


# ── Request models ────────────────────────────────────────────────────────────

class RepoSetupRequest(BaseModel):
    github_url: str   # e.g. "https://github.com/owner/repo" or "owner/repo"


class ChatRequest(BaseModel):
    message: str


# ── Repo indexing ─────────────────────────────────────────────────────────────

def _parse_repo(github_url: str) -> str:
    """Extract 'owner/repo' from a full GitHub URL or a bare 'owner/repo' string."""
    url = github_url.rstrip("/")
    if "github.com" in url:
        parts = url.split("github.com/")[-1].split("/")
        return f"{parts[0]}/{parts[1]}"
    return url  # already in owner/repo format


def _run_indexing(github_repo: str):
    global _indexing_status
    try:
        _indexing_status = {
            "status": "indexing",
            "repo": github_repo,
            "message": f"Cloning and indexing {github_repo}…",
        }
        os.environ["GITHUB_REPO"] = github_repo

        from rag.indexer import index_repo
        index_repo(github_repo=github_repo)

        _indexing_status = {
            "status": "ready",
            "repo": github_repo,
            "message": f"{github_repo} is indexed and ready.",
        }
    except Exception as e:
        _indexing_status = {
            "status": "error",
            "repo": github_repo,
            "message": str(e),
        }


@app.post("/repo/setup")
def setup_repo(req: RepoSetupRequest):
    github_repo = _parse_repo(req.github_url)
    t = threading.Thread(target=_run_indexing, args=(github_repo,), daemon=True)
    t.start()
    return {"message": f"Indexing started for {github_repo}."}


@app.get("/repo/status")
def repo_status():
    return _indexing_status


# ── Chat ──────────────────────────────────────────────────────────────────────

def _build_history_context() -> str:
    """Summarise the last 4 turns into a context block for the planner."""
    if not _chat_history:
        return ""
    recent = _chat_history[-4:]
    lines = []
    for turn in recent:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
        if turn.get("target_file"):
            lines.append(f"(last changed file: {turn['target_file']})")
    return "\n".join(lines)


@app.post("/chat")
def chat(req: ChatRequest):
    if _indexing_status["status"] not in ("ready",):
        raise HTTPException(
            status_code=400,
            detail="No repo is ready. Index a repo first via POST /repo/setup."
        )

    # Inject conversation history so follow-up messages work
    history_ctx = _build_history_context()
    full_task = req.message
    if history_ctx:
        full_task = (
            f"Previous conversation:\n{history_ctx}\n\n"
            f"New request: {req.message}"
        )

    from graph.graph import build_graph
    graph = build_graph()

    try:
        final_state = graph.invoke({
            "task": full_task,
            "subtasks": [],
            "retrieved_context": "",
            "generated_code": "",
            "diff": "",
            "target_file": None,
            "error": None,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    diff = final_state.get("diff", "")
    generated_code = final_state.get("generated_code", "")
    target_file = final_state.get("target_file") or "unknown"
    error = final_state.get("error")

    if error:
        raise HTTPException(status_code=500, detail=error)

    message_id = str(uuid.uuid4())
    _pending[message_id] = {
        "target_file": target_file,
        "generated_code": generated_code,
        "task": req.message,
    }

    assistant_msg = (
        f"Generated changes for `{target_file}`. "
        "Review the diff on the right and click **Approve & Push PR** to open a pull request."
    )

    turn = {
        "id": message_id,
        "timestamp": datetime.now().isoformat(),
        "user": req.message,
        "assistant": assistant_msg,
        "diff": diff,
        "target_file": target_file,
        "pr_url": None,
    }
    _chat_history.append(turn)
    _save_history()

    return {
        "message_id": message_id,
        "assistant": assistant_msg,
        "diff": diff,
        "target_file": target_file,
    }


@app.post("/chat/approve/{message_id}")
def approve_pr(message_id: str):
    if message_id not in _pending:
        raise HTTPException(status_code=404, detail="No pending change for this message.")

    pending = _pending.pop(message_id)
    os.environ["GITHUB_REPO"] = _indexing_status.get("repo", os.environ.get("GITHUB_REPO", ""))

    from agents.pr_agent import create_github_pr
    result = create_github_pr(
        target_file=pending["target_file"],
        generated_code=pending["generated_code"],
        task_description=pending["task"],
    )

    pr_url = result if result.startswith("SUCCESS") else None
    for turn in reversed(_chat_history):
        if turn.get("id") == message_id:
            turn["pr_url"] = result
            break
    _save_history()

    return {"result": result, "pr_url": pr_url}


@app.get("/chat/history")
def get_history():
    return _chat_history


@app.delete("/chat/history")
def clear_history():
    global _chat_history, _pending
    _chat_history = []
    _pending = {}
    _save_history()
    return {"message": "History cleared."}


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found in web/ folder.")
    return html_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
