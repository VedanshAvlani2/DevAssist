from crewai import Agent
from crewai.tools import tool
from rag.retriever import get_relevant_context

@tool("RAG Retriever Tool")
def rag_retriever_tool(query: str) -> str:
    '''
    Searches the indexed codebase using semantic similarity.
    Input: a natural language query describing what code to find.
    Output: relevant code chunks from ChromaDB.
    '''
    results = get_relevant_context(query)
    return results

def make_retriever_agent():
    return Agent(
        role="Code Retriever",
        goal=(
            "Search the indexed codebase and return the most relevant code chunks "
            "for the given task. Use the RAG Retriever Tool to find context."
        ),
        backstory=(
            "You are a codebase search specialist. "
            "You use semantic search to find the exact code snippets "
            "that are most relevant to a given coding task."
        ),
        tools=[rag_retriever_tool],
        verbose=True,
        allow_delegation=False,
    )
