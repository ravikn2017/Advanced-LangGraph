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

# --- Graph nodes ---
def assistant(state: MessagesState):
  """LLM node: reads message history and returns the next AI message (may include tool calls)."""
  return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build the graph: assistant ↔ tools loop (same pattern as ReAct agent)
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
  "assistant",
  tools_condition,  # tool call in last message → "tools"; otherwise → END
)
builder.add_edge("tools", "assistant")  # after tool runs, go back to assistant

# Checkpointer saves graph state after each step (required for interrupts + resume)
memory = MemorySaver()

# interrupt_before=["assistant"] → graph STOPS before assistant runs, waiting for human
hitl_graph = builder.compile(interrupt_before=["assistant"], checkpointer=memory)

# Optional: visualize the graph
hitl_graph.get_graph().draw_mermaid_png(output_file_path="HITL-graph.png")


# =============================================================================
# DEMO 1: Basic HITL — pause, inspect, then resume
# =============================================================================
# thread_id identifies a separate conversation in the checkpointer
thread: RunnableConfig = {"configurable": {"thread_id": "123"}}
initial_input: MessagesState = {
    "messages": [HumanMessage(content="Multiply 2 and 3")]
}

# Step 1: Start the graph. Stops at interrupt BEFORE assistant (only human message shown)
print("\n--- Demo 1: Step 1 — graph paused before assistant ---")
for event in hitl_graph.stream(initial_input, thread, stream_mode="values"):
  event["messages"][-1].pretty_print()

# Inspect checkpoint: state.next shows which node will run next (should be 'assistant')
state = hitl_graph.get_state(thread)
print(f"Next node to run: {state.next}")

# Step 2: Resume with stream(None, ...) — human approved, assistant runs + may call tools
print("\n--- Demo 1: Step 2 — resume after human approval ---")
for event in hitl_graph.stream(None, thread, stream_mode="values"):
  event["messages"][-1].pretty_print()

# Step 3: Resume again — completes tool execution and final AI reply
print("\n--- Demo 1: Step 3 — finish tool loop and get final answer ---")
for event in hitl_graph.stream(None, thread, stream_mode="values"):
  event["messages"][-1].pretty_print()


# =============================================================================
# DEMO 2: Human edits the request before assistant runs
# =============================================================================
# Use a NEW thread_id so this demo doesn't mix with Demo 1's completed state
edit_thread: RunnableConfig = {"configurable": {"thread_id": "1"}}
edit_input: MessagesState = {
    "messages": [HumanMessage(content="Multiply 2 and 3")]
}

# Step 1: Start and pause before assistant (same as Demo 1)
print("\n--- Demo 2: Step 1 — start and pause before assistant ---")
for event in hitl_graph.stream(edit_input, edit_thread, stream_mode="values"):
  event["messages"][-1].pretty_print()

# Step 2: Human corrects the request BEFORE assistant runs
# update_state APPENDS the new message to the conversation (add_messages reducer)
print("\n--- Demo 2: Step 2 — human edits input via update_state ---")
hitl_graph.update_state(
  edit_thread,
  {"messages": [HumanMessage(content="No, please multiply 15 and 6")]},
)

# Show full message history after the edit (original + correction)
new_state = hitl_graph.get_state(edit_thread).values
for m in new_state["messages"]:
  m.pretty_print()

# Step 3+: Resume with None on edit_thread (NOT edit_input — that would re-send "Multiply 2 and 3")
# IMPORTANT: always use the same thread_id you edited; don't use `thread` from Demo 1
print("\n--- Demo 2: Step 3+ — resume with corrected input ---")
for event in hitl_graph.stream(None, edit_thread, stream_mode="values"):
  event["messages"][-1].pretty_print()

for event in hitl_graph.stream(None, edit_thread, stream_mode="values"):
  event["messages"][-1].pretty_print()

## Workflow will wait for the User Input