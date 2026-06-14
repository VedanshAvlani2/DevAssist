from langgraph.graph import StateGraph, END
from graph.state import DevAssistState
from graph.nodes.planner import plan_task
from graph.nodes.retriever import retrieve_context
from graph.nodes.generator import generate_code_node
from graph.nodes.differ import make_diff

def build_graph():
    builder = StateGraph(DevAssistState)
    builder.add_node("plan_task", plan_task)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("generate_code", generate_code_node)
    builder.add_node("make_diff", make_diff)
    builder.set_entry_point("plan_task")
    builder.add_edge("plan_task", "retrieve_context")
    builder.add_edge("retrieve_context", "generate_code")
    builder.add_edge("generate_code", "make_diff")
    builder.add_edge("make_diff", END)
    return builder.compile()

if __name__ == "__main__":
    TEST_TASK = "Add a docstring to every function in rag/retriever.py"
    graph = build_graph()
    print("=" * 60)
    print(f"Running DevAssist graph on task:")
    print(f"  {TEST_TASK}")
    print("=" * 60)
    final_state = graph.invoke({
        "task": TEST_TASK,
        "subtasks": [],
        "retrieved_context": "",
        "generated_code": "",
        "diff": "",
        "target_file": None,
        "error": None
    })
    print("\n" + "=" * 60)
    print("FINAL DIFF OUTPUT:")
    print("=" * 60)
    print(final_state["diff"])
