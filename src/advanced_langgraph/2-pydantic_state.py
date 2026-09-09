from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel


class State(BaseModel):
    name: str


def example_node(state: State):
    return {"name": "Hello"}


builder = StateGraph(State)
builder.add_node("example_node", example_node)
builder.add_edge(START, "example_node")
builder.add_edge("example_node", END)

graph = builder.compile()

result = graph.invoke(State(name="Ravi"))
print(result)
