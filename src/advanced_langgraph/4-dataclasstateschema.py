from typing_extensions import NotRequired, TypedDict
from typing import Literal
import random
from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass

#Data Classes
# Python's dataclasses provide another way to define structured data
# Dataclasses offer a concise syntax for crating classes that are primarily used to store data

@dataclass
class DataClassState:
  name: str
  game:Literal["badminton","cricket"]

class TypedDictState(TypedDict):
  name: str
  game: NotRequired[Literal["cricket", "badminton"]]

# def play_game(state:TypedDictState):
#   print("---Play Game node has been called--")
#   return {"name":state['name'] + " want to play "}

def play_game(state:DataClassState):
  print("---Play Game node has been called--")
  return {"name":state.name + " want to play "}

# def decide_play(state:TypedDictState)->Literal["cricket","badminton"]:
#   if random.random() < 0.5:
#     return "cricket"
#   else:
#     return "badminton"
def decide_play(state:DataClassState)->Literal["cricket","badminton"]:
  if random.random() < 0.5:
    return "cricket"
  else:
    return "badminton"

# def cricket(state:TypedDictState):
#   print("--Cricket node has been called--")
#   return {"name":state["name"] + "cricket", "game":"cricket"}

# def badminton(state:TypedDictState):
#   print("--Badminton node has been called--")
#   return {"name":state["name"] + "badminton", "game":"badminton"}
def cricket(state:DataClassState):
  print("--Cricket node has been called--")
  return {"name":state.name + "cricket", "game":"cricket"}

def badminton(state:DataClassState):
  print("--Badminton node has been called--")
  return {"name":state.name + "badminton", "game":"badminton"}

# builder=StateGraph(TypedDictState)
builder=StateGraph(DataClassState)
builder.add_node("playgame",play_game)
builder.add_node("cricket",cricket)
builder.add_node("badminton",badminton)

## Flow of the graph
builder.add_edge(START,"playgame")
builder.add_conditional_edges("playgame",decide_play)
builder.add_edge("cricket",END)
builder.add_edge("badminton",END)

# Add
graph = builder.compile()

# View
#graph.get_graph().draw_mermaid_png(output_file_path="dataclassstate-graph.png")

# Invocation
# result = graph.invoke({"name": "Ravi"})
result = graph.invoke(DataClassState(name="Ravi",game="badminton"))
print(result)


