from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

llm=ChatOpenAI(model="gpt-4o")
result=llm.invoke("Hello")
print(result)

#Graph State
class State(TypedDict):
  topic: str
  characters: str
  settings: str
  premises: str
  story_intro: str

## Nodes
def generate_characters(state:State):
  """Generate character descriptions"""
  msg=llm.invoke(f"Create two character name and brief traits for a story about {state["topic"]}")
  return {"characters":msg.content}

def generate_setting(state:State):
  """Generate a story setting"""
  msg=llm.invoke(f"Describe a vivid setting for a story about {state["topic"]}")
  return {"settings":msg.content}

def generate_premise(state:State):
  """Generate a story premise"""
  msg=llm.invoke(f"Write a one-sentence plot premise for a story about {state["topic"]}")
  return {"premises":msg.content}

def combine_elements(state: State):
  """Combine characters, setting, and premise into an intro"""
  msg = llm.invoke(
    f"Write a short story introduction using these elements:\n"
    f"Characters: {state['characters']}\n"
    f"Setting: {state['settings']}\n"
    f"Premise: {state['premises']}"
  )
  return {"story_intro":msg.content}

#Build Graph
builder = StateGraph(State)
builder.add_node("character", generate_characters)
builder.add_node("setting", generate_setting)
builder.add_node("premise", generate_premise)
builder.add_node("combine", combine_elements)

#Define Edges (parallel execution from START)
builder.add_edge(START, "character")
builder.add_edge(START, "setting")
builder.add_edge(START, "premise")
builder.add_edge("character", "combine")
builder.add_edge("setting", "combine")
builder.add_edge("premise", "combine")
builder.add_edge("combine", END)

#Compile and Run
compiled_graph = builder.compile()

#View
compiled_graph.get_graph().draw_mermaid_png(output_file_path="Parallelization-graph.png")

input_state: State = {
    "topic": "a detective solving mysteries in Victorian London",
    "characters": "",
    "settings": "",
    "premises": "",
    "story_intro": "",
}
result = compiled_graph.invoke(input_state)
print(result["story_intro"])