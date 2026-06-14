# mcp/server.py

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import asyncio
import sys
import os

# Add mcp/ folder to path so tools/ imports work
sys.path.append(os.path.dirname(__file__))

from tools.retrieve import retrieve_code
from tools.generate import generate_code

# Create MCP server instance
app = Server("devassist")


# ── Register Tools ──────────────────────────────────────────────


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """Tell clients what tools are available."""
    return [
        types.Tool(
            name="retrieve_code",
            description=(
                "Search the indexed GitHub repository for code chunks "
                "relevant to a natural language query. "
                "Use this before generating code to get context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what code to find",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of chunks to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="generate_code",
            description=(
                "Generate Python code using the fine-tuned Mistral 7B model. "
                "Optionally pass context from retrieve_code for better results."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Natural language task description",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional: code context from retrieve_code",
                        "default": "",
                    },
                },
                "required": ["instruction"],
            },
        ),
    ]


# ── Handle Tool Calls ───────────────────────────────────────────


@app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent]:
    """Route incoming tool calls to the right function."""

    if name == "retrieve_code":
        query = arguments["query"]
        n_results = arguments.get("n_results", 5)
        result = await asyncio.to_thread(retrieve_code, query, n_results)
        return [types.TextContent(type="text", text=result)]

    elif name == "generate_code":
        instruction = arguments["instruction"]
        context = arguments.get("context", "")
        result = await asyncio.to_thread(generate_code, instruction, context)
        return [types.TextContent(type="text", text=result)]

    else:
        return [
            types.TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )
        ]


# ── Entry Point ─────────────────────────────────────────────────


async def main():
    """Start MCP server on stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())