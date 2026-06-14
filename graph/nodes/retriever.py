# graph/nodes/retriever.py
# This node calls the MCP server's retrieve_code tool.
# It uses the task + subtasks to build a search query,
# then fetches matching code chunks from ChromaDB.

import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from graph.state import DevAssistState


async def _call_mcp_retrieve(query: str) -> str:
    """
    Opens a stdio connection to our MCP server,
    calls the retrieve_code tool, returns the result string.
    """
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server/server.py"],  # path to your MCP server
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "retrieve_code",
                arguments={"query": query}
            )

            # result.content is a list of ContentBlock objects
            # Each has a .text attribute
            texts = [block.text for block in result.content if hasattr(block, "text")]
            return "\n\n".join(texts)


def retrieve_context(state: DevAssistState) -> DevAssistState:
    """
    Node 2: Retrieve
    Input:  state["task"], state["subtasks"]
    Output: state["retrieved_context"]
    """

    # Build a rich query from task + all subtasks
    query_parts = [state["task"]] + state.get("subtasks", [])
    query = " ".join(query_parts)

    print(f"[retriever] Query: {query[:100]}...")

    # Run async MCP call from sync context
    context = asyncio.run(_call_mcp_retrieve(query))

    print(f"[retriever] Retrieved {len(context)} chars of context")

    return {
        **state,
        "retrieved_context": context
    }