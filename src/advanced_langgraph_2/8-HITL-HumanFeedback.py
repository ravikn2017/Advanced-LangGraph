"""
Human-in-the-Loop (HITL) demo with LangGraph.

Flow overview:
  1. User sends a message
  2. Graph pauses BEFORE the assistant runs (interrupt_before)
  3. Human can inspect state, edit messages, then resume
  4. Assistant may call tools, then reply with the final answer

Key APIs used:
  - interrupt_before=["assistant"]  → pause for human approval
  - MemorySaver + thread_id         → persist conversation per thread
  - stream(None, config)              → resume from last checkpoint
  - update_state(config, {...})       → human edits state before resuming
"""

import operator
from pathlib import Path
from typing import Annotated, List, Literal, cast
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.types import Send
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

load_dotenv()

llm_openai = ChatOpenAI(model="gpt-4o")

# --- Tools the assistant can call ---
def multiply(a: int, b: int) -> int:
  """Multiply a and b."""
  return a * b

def add(a: int, b: int) -> int:
  """Adds a and b."""
  return a + b

def divide(a: int, b: int) -> float:
  """Divide a and b."""
  return a / b

tools = [add, multiply, divide]

# Bind tools so the LLM can decide when to call them
llm_with_tools = llm_openai.bind_tools(tools)

# System prompt prepended to every assistant call
sys_msg = SystemMessage(
  content="You are a helpful assistant tasked with performing arithmetic on a set of inputs"
)

## Human Feedback Node
def human_feedback(state:MessagesState):
  pass

## Assistant Node
def assistant(state:MessagesState):
  return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

## Graph
builder = StateGraph(MessagesState)

## Define Nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("human_feedback", human_feedback)

## Define the edges
builder.add_edge(START, "human_feedback")
builder.add_edge("human_feedback", "assistant")
builder.add_conditional_edges(
  "assistant",
  tools_condition,
)
builder.add_edge("tools","human_feedback")

memory=MemorySaver()
hitl_graph=builder.compile(interrupt_before=["human_feedback"],checkpointer=memory)
hitl_graph.get_graph().draw_mermaid_png(output_file_path="HITL-HumanFeedback-graph.png")

# Input — messages must be a list of message objects, not a plain string
initial_input: MessagesState = {
    "messages": [HumanMessage(content="Multiply 2 and 3")]
}

# Thread — required when using a checkpointer (MemorySaver)
thread: RunnableConfig = {"configurable": {"thread_id": "5"}}

# Run the graph until the first interruption
print("\n--- Demo 1: Step 1 — graph paused before Human Feedback ---")
for event in hitl_graph.stream(initial_input, thread, stream_mode="values"):
  event["messages"][-1].pretty_print()

## Get User Input
user_input=input("Tell me how you want to update the state:")
hitl_graph.update_state(
  thread,
  {"messages": user_input}, 
  as_node="human_feedback",
)

## Continue the graph execution
for event in hitl_graph.stream(None, thread, stream_mode="values"):
  event["messages"][-1].pretty_print()

for event in hitl_graph.stream(None, thread, stream_mode="values"):
  event["messages"][-1].pretty_print()
