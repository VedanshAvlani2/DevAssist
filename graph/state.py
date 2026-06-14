# graph/state.py
# This file defines the "memory" that flows between every node in the graph.
# Each field gets populated as the graph runs, then passed forward.

from typing import TypedDict, Optional, List

class DevAssistState(TypedDict):
    # INPUT: natural language task from the user
    task: str

    # OUTPUT of plan_task node: list of subtask strings
    subtasks: List[str]

    # OUTPUT of retrieve_context node: raw code chunks from ChromaDB
    retrieved_context: str

    # OUTPUT of generate_code node: the new/modified code
    generated_code: str

    # OUTPUT of make_diff node: unified diff string
    diff: str

    # The file path being modified (parsed from the plan)
    target_file: Optional[str]

    # Any error message if a node fails
    error: Optional[str]