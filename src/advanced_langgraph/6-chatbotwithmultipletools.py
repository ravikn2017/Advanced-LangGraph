## Building Chatbot with Multiple tools using Langgraph
# Create a chatbot with tool capabilities from arxiv and tavily search

from dotenv import load_dotenv
from langchain_arxiv import ArxivRetriever
from langchain_arxiv.exceptions import ArxivAPIError
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

arxiv_retriever = ArxivRetriever(k=2, max_content_chars=500)


@tool
def arxiv(query: str) -> str:
    """Search arxiv.org for scientific papers. Input should be a search query."""
    try:
        docs = arxiv_retriever.invoke(query)
    except ArxivAPIError:
        return (
            "arXiv rate limit reached (HTTP 429). "
            "Wait 30-60 seconds and try again."
        )
    if not docs:
        return "No good Arxiv Result was found"
    return "\n\n".join(
        f"Published: {doc.metadata['updated']}\n"
        f"Title: {doc.metadata['title']}\n"
        f"Authors: {', '.join(doc.metadata['authors'])}\n"
        f"Summary: {doc.page_content}"
        for doc in docs
    )


tavily_tool = TavilySearch(max_results=1)
tools = [arxiv, tavily_tool]

llm_openai = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm_openai.bind_tools(tools)

#State Schema
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from typing import Annotated
from langgraph.graph.message import add_messages
class State(TypedDict):
  messages:Annotated[list[AnyMessage],add_messages]


#Entire Chatbot with LangGraph
from langgraph.graph import StateGraph, START,END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# Node Definition
def tool_calling_llm(state:State):
  return {"messages":[llm_with_tools.invoke(state["messages"])]}

# Build Graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
  "tool_calling_llm",
  # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
  # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
  tools_condition
)
builder.add_edge("tools", END)

graph = builder.compile()

if __name__ == "__main__":
    print("=== Arxiv Search ===")
    print(arxiv.invoke("Attention is all you need"))

    print("\n=== Tavily Search ===")
    print(tavily_tool.invoke("Recent AI news"))

    print("\n=== LangGraph Chatbot ===")
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is attention all you need")]}
    )
    for message in result["messages"]:
        message.pretty_print()

    print("\nDone.")