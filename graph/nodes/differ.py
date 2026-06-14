# graph/nodes/differ.py
# This node compares the original file content with the generated code.
# It produces a unified diff — the same format used in GitHub PRs.
# If no target_file was identified, it diffs against an empty original.

import difflib
import os
from graph.state import DevAssistState


def make_diff(state: DevAssistState) -> DevAssistState:
    """
    Node 4: Diff
    Input:  state["target_file"], state["generated_code"]
    Output: state["diff"]
    """

    target_file = state.get("target_file")
    generated_code = state.get("generated_code", "")

    # Read original file if it exists
    if target_file and os.path.isfile(target_file):
        with open(target_file, "r", encoding="utf-8") as f:
            original_lines = f.readlines()
        original_label = target_file
    else:
        # No original file — treat as new file creation
        original_lines = []
        original_label = "/dev/null"

    # Split generated code into lines for difflib
    generated_lines = [line + "\n" for line in generated_code.splitlines()]

    # Generate unified diff (same format as `git diff`)
    diff_lines = difflib.unified_diff(
        original_lines,
        generated_lines,
        fromfile=f"a/{original_label}",
        tofile=f"b/{target_file or 'generated.py'}",
        lineterm=""
    )

    diff_output = "\n".join(diff_lines)

    if not diff_output.strip():
        diff_output = "# No changes detected between original and generated code."

    print(f"[differ] Diff produced ({len(diff_output)} chars)")

    return {
        **state,
        "diff": diff_output
    }