"""
DevAssist Eval Harness
======================
Runs 6 eval tasks through the LangGraph pipeline and scores each result
using GPT-4o as a judge (1–5 scale). No PRs are created — diffs are
evaluated directly, so you can run this as many times as you like cleanly.

Prerequisites
-------------
1. ChromaDB HTTP server running in venv (Python 3.14):
       chroma run --path ./chroma_db --port 8000

2. retrylib indexed in venv312:
       GITHUB_REPO=VedanshAvlani2/retrylib python -m rag.indexer

3. OPENAI_API_KEY set in .env

Run
---
    python -m evals.run_evals
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Point the retriever at the retrylib collection for all eval tasks
os.environ["GITHUB_REPO"] = "VedanshAvlani2/retrylib"

from openai import OpenAI
from graph.graph import build_graph

TASKS_PATH = os.path.join(os.path.dirname(__file__), "tasks.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "eval_report.json")


# ── Judge ────────────────────────────────────────────────────────────────────

def judge_output(task: dict, generated_code: str, diff: str) -> dict:
    """
    Use GPT-4o to score the generated code on a 1–5 scale.

    5 = Correctly and cleanly fixes the issue
    4 = Mostly correct, minor issues
    3 = Partially addresses the task
    2 = Attempted but incorrect fix
    1 = No meaningful change or wrong direction
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = f"""You are evaluating an AI coding assistant's output.

Task description:
{task['description']}

The fix should contain or relate to: "{task['expected_keyword']}"

Generated diff:
{diff[:3000] if diff else "(no diff produced)"}

Score the output 1–5 using this rubric:
5 = Correctly and cleanly fixes the described issue
4 = Mostly correct with minor issues (e.g., style, extra changes)
3 = Partially addresses the task but misses something important
2 = Attempted a fix but it is logically incorrect
1 = No meaningful change, completely wrong direction, or empty output

Respond with JSON only: {{"score": <1-5>, "reason": "<one concise sentence>"}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


# ── Runner ───────────────────────────────────────────────────────────────────

def run_evals():
    with open(TASKS_PATH) as f:
        tasks = json.load(f)

    graph = build_graph()
    results = []

    print(f"\n{'=' * 62}")
    print(f"  DevAssist Eval Harness — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Repo : {os.environ['GITHUB_REPO']}")
    print(f"  Tasks: {len(tasks)}")
    print(f"{'=' * 62}\n")

    for task in tasks:
        print(f"[{task['id']}] {task['category'].upper()}")
        print(f"  {task['description'][:80]}...")

        start = time.time()
        status = "FAIL"
        score_data = {"score": 0, "reason": "Unknown error."}

        try:
            final_state = graph.invoke({
                "task": task["description"],
                "subtasks": [],
                "retrieved_context": "",
                "generated_code": "",
                "diff": "",
                "target_file": None,
                "error": None,
            })

            diff = final_state.get("diff", "")
            generated_code = final_state.get("generated_code", "")
            elapsed = round(time.time() - start, 1)

            if not generated_code or not generated_code.strip():
                score_data = {"score": 1, "reason": "Pipeline produced no code."}
                status = "FAIL"
            else:
                score_data = judge_output(task, generated_code, diff)
                status = "PASS" if score_data["score"] >= 3 else "FAIL"

        except Exception as e:
            elapsed = round(time.time() - start, 1)
            score_data = {"score": 0, "reason": str(e)}
            status = "ERROR"
            diff = ""

        result = {
            "id": task["id"],
            "category": task["category"],
            "status": status,
            "score": score_data["score"],
            "reason": score_data["reason"],
            "elapsed_s": elapsed,
        }
        results.append(result)

        icon = "✓" if status == "PASS" else ("✗" if status == "FAIL" else "!")
        print(f"  {icon} {status}  |  Score: {score_data['score']}/5  |  {elapsed}s")
        print(f"    {score_data['reason']}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["status"] == "PASS")
    errored = sum(1 for r in results if r["status"] == "ERROR")
    scored = [r["score"] for r in results if r["score"] > 0]
    avg_score = round(sum(scored) / len(scored), 1) if scored else 0.0

    print(f"{'=' * 62}")
    print(f"  PASSED : {passed}/{len(tasks)}")
    print(f"  ERRORS : {errored}")
    print(f"  AVG SCORE: {avg_score}/5")
    print(f"{'=' * 62}\n")

    # ── Save report ──────────────────────────────────────────────────────────
    report = {
        "timestamp": datetime.now().isoformat(),
        "repo": os.environ["GITHUB_REPO"],
        "passed": passed,
        "total": len(tasks),
        "errored": errored,
        "avg_score": avg_score,
        "results": results,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved → evals/eval_report.json")
    return report


if __name__ == "__main__":
    run_evals()
