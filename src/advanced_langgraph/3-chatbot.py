## Implementing simple chatbot using Langgraph
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(model="gpt-4o")
llm_ollama = ChatOllama(model="gemma4:latest", temperature=0)


def superbot(state: State):
    return {"messages": [llm_ollama.invoke(state["messages"])]}


graph = StateGraph(State)
graph.add_node("SuperBot", superbot)
graph.add_edge(START, "SuperBot")
graph.add_edge("SuperBot", END)

compiled_graph = graph.compile()
compiled_graph.get_graph().draw_mermaid_png(output_file_path="chatbot-graph.png")

## Invocation
result = compiled_graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="Hi My name is Ravi and I like Football and Formula one"
            )
        ]
    }
)
print(result["messages"][-1].content)

# Streaming the responses
for event in compiled_graph.stream(
    {"messages": [HumanMessage(content="Hello My name is Ravi")]},
    stream_mode="values",
):
    print(event)