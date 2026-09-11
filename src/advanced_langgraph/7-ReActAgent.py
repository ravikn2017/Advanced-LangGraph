import os
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "ReAct Agent"

## Custom Functions
@tool
def multiply(a: int, b: int) -> int:
  """Multiply a and b."""
  return a * b

@tool
def add(a: int, b: int) -> int:
  """Adds a and b."""
  return a + b

@tool
def divide(a: int, b: int) -> float:
  """Divide a and b."""
  return a / b

#Combine all the tools in the list
tools=[add,multiply,divide]

# Initialize LLM Model
llm_openai = init_chat_model("gpt-4o", model_provider="openai")
llm_with_tools = llm_openai.bind_tools(tools)

# result=llm_with_tools.invoke([HumanMessage(content=f"What is 2 plus 3 and get product also")])
# print(result)

## State Schema
class State(TypedDict):
  messages:Annotated[list[AnyMessage],add_messages]

## Node definition
def tool_calling_llm(state: State):
  return {"messages": [llm_with_tools.invoke(state["messages"])]}

## Build graph
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
builder.add_edge("tools", "tool_calling_llm")

graph = builder.compile()

#View
#graph.get_graph().draw_mermaid_png(output_file_path="ReActAgent-graph.png")

print("\n=== LangGraph Chatbot ===")
result = graph.invoke(
    {"messages": [HumanMessage(content="Add 5 plus 5 and then multiply 2 and 7")]}
)
for message in result["messages"]:
    message.pretty_print()

result = graph.invoke(
    {"messages": [HumanMessage(content="Divide that by 5")]}
)
for message in result["messages"]:
    message.pretty_print()

# Memory Saver
# LangGraph can use a checkpointer to automatically save the graph state after each step
# This build-in persistence layer gives us  memory, allowing LangGraph to pick up from the last
# state update. One of the easiest checkpointers to use is the MemorySaver, an in-memory 
# key-value store for Graph State.
# All we need to do is simply compile the graph with a checkpointer, and our graph has memory!

memory=MemorySaver()
graph_memory = builder.compile(checkpointer=memory)

# Specify the thread

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

# Specify an input
input_messages: list[AnyMessage] = [HumanMessage(content="Add 12 and 13")]
result_memory = graph_memory.invoke({"messages": input_messages}, config=config)
for m in result_memory["messages"]:
    m.pretty_print()

# The next input message should be able to take info from previous response
input_messages2: list[AnyMessage] = [HumanMessage(content="divide that number by 5")]
result_memory2 = graph_memory.invoke({"messages": input_messages2}, config=config)
for m in result_memory2["messages"]:
    m.pretty_print()

print("\nDone.")