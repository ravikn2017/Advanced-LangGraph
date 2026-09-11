import asyncio
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()


class State(TypedDict):
  messages: Annotated[list[AnyMessage], add_messages]

llm=ChatOpenAI(model="gpt-4o")
llm.invoke("Hello")

#Create Nodes
memory=MemorySaver()
def superbot(state:State):
  return {"messages":[llm.invoke(state['messages'])]}

graph=StateGraph(State)

## Node
graph.add_node("SuperBot", superbot)

## Edges
graph.add_edge(START, "SuperBot")
graph.add_edge("SuperBot", END)

graph_builder=graph.compile(checkpointer=memory)

#View
#graph_builder.get_graph().draw_mermaid_png(output_file_path="streaming-graph.png")

#Invocation
config: RunnableConfig = {"configurable": {"thread_id": "1"}}
input_messages: list[AnyMessage] = [
    HumanMessage(content="Hi, My name is Ravi and I like football")
]
result = graph_builder.invoke({"messages": input_messages}, config=config)
print(result)

#Create another thread
stream_config: RunnableConfig = {"configurable": {"thread_id": "2"}}
stream_messages: list[AnyMessage] = [
    HumanMessage(content="Hi My name is Ravi and I like football")
]

for chunk in graph_builder.stream(
    {"messages": stream_messages}, stream_config, stream_mode="updates"
):
    print(chunk)

for chunk in graph_builder.stream(
    {"messages": stream_messages}, stream_config, stream_mode="values"
):
    print(chunk)

# Astream Method
events_config: RunnableConfig = {"configurable": {"thread_id": "3"}}
events_messages: list[AnyMessage] = [
    HumanMessage(content="Hi My Name is Krish and I like to play cricket")
]


async def stream_events() -> None:
    async for event in graph_builder.astream_events(
        {"messages": events_messages}, events_config, version="v2"
    ):
        print(event)


asyncio.run(stream_events())