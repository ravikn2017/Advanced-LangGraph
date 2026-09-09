#Chains using LangGraph
# 4 Important Concepts
# 1. Use Chat Messages in graph nodes
# 2. Use Chat Models in graph nodes
# 3. How to bind tools to our LLM in chat models
# 4. How to execute the tools call in our graph nodes

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, MessageLikeRepresentation
from langchain_core.tools import tool
load_dotenv()

import os

# How to use chat messages as our graph state
# We can use message which can be used to capture different roles within a conversation.
# LangChain has various message types including HumanMessage, AIMessage, SystemMessage and ToolMessage
# These represent a message from the user, from chat model, for the chat model to instruct behaviour
# and from a tool call

# Every message have these important components
# - content - content of the message
# - name - Specify the name of author
# response_metadata - optionally, a dict of metadata (Ex - often populated by model provider for AIMessage)

messages: list[BaseMessage] = [
    AIMessage(content="Please tell me how can I help", name="LLMModel")
]
messages.append(HumanMessage(content="I want to learn coding",name="Ravi"))
messages.append(AIMessage(content=f"Which programming language you want to learn",name="LLMModel"))

for message in messages:
  message.pretty_print()

#Chat Models
#Use the sequence of message as iinput with the chatmodels using LLMs and OPENAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
llm_ollama=ChatOllama(model="gemma4:latest")
llm_openai=ChatOpenAI(model="gpt-4o")

result=llm_openai.invoke(messages)
print(result)

result.response_metadata

#Tools
#Tools can be integrated with the LLM models to integrate with external systems.
#External systems can be APIs, third-party tools
#Whenever a query is asked the model can choose to call the tool and this query is based 
#on the natural language input and this will return an output that matches the tool's schema

@tool
def add(a: int, b: int) -> int:
  """Add a and b."""
  return a + b

tools = [add]

## Binding tool with llm
llm_with_tools = llm_openai.bind_tools(tools)
tool_result = llm_with_tools.invoke([HumanMessage(content="What is 3 plus 5?",name="Ravi")])
print(tool_result.tool_calls)

## Using messages as state
from typing import Annotated

from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
  messages:Annotated[list[AnyMessage], add_messages]

#Reduces with add_messages

initial_messages: list[MessageLikeRepresentation] = [
    AIMessage(content="Please tell me how can I help", name="LLMModel")
]
initial_messages.append(HumanMessage(content="I want to learn coding", name="Ravi"))
print(initial_messages)

ai_message = AIMessage(
    content="Which programming language yo want to learn", name="LLMModel"
)
print(ai_message)

#Reducers add_messages is to append instead of override
updated_messages = add_messages(initial_messages, ai_message)
print(updated_messages)

#chatbot node functionality
def llm_tool(state:State):
  return {"messages":[llm_with_tools.invoke(state["messages"])]}

from langgraph.graph import StateGraph, START, END

# builder=StateGraph(State)
# builder.add_node("llm_tool", llm_tool)

# builder.add_edge(START,"llm_tool")
# builder.add_edge("llm_tool",END)

# graph=builder.compile()

#graph.get_graph().draw_mermaid_png(output_file_path="chainsLang-graph.png")
# Invocation
# graph_result = graph.invoke(
#     {"messages": [HumanMessage(content="What is 2 plus 2")]}
# )

# for message in graph_result["messages"]:
#     message.pretty_print()

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

builder=StateGraph(State)

## Add Nodes
builder.add_node("llm_tool",llm_tool)
builder.add_node("tools",ToolNode(tools))

## Add Edge
builder.add_edge(START,"llm_tool")
builder.add_conditional_edges(
  "llm_tool",
  # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
  # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
  tools_condition
)
builder.add_edge("tools",END)

graph_builder=builder.compile()
graph_builder.get_graph().draw_mermaid_png(output_file_path="chainsLang-graph.png")

# Invocation
graph_result = graph_builder.invoke(
    {"messages": [HumanMessage(content="What is 2 plus 2")]}
)

for message in graph_result["messages"]:
    message.pretty_print()

# Invocation with a different message
graph_result = graph_builder.invoke(
    {"messages": [HumanMessage(content="What is Machine Learning")]}
)

for message in graph_result["messages"]:
    message.pretty_print()