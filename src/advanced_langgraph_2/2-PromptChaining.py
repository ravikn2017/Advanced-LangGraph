from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

llm_openai = ChatOpenAI(model="gpt-4o")
result=llm_openai.invoke("Hello")
print(result)

#Graph State
class State(TypedDict):
  topic:str
  story:str
  improved_story:str
  final_story:str

## Nodes
def generate_story(state:State):
  msg=llm_openai.invoke(f"Write a one sentence story premise about {state["topic"]}")
  return {"story":msg.content}

def check_conflict(state:State):
  if "?" in state["story"] or "!" in state["story"]:
    return "Fail"
  return "Pass"

def improved_story(state:State):
  msg=llm_openai.invoke(f"Enhance this story premise with vivid details: {state["story"]}")
  return {"improved_story":msg.content}

def polish_story(state:State):
  msg=llm_openai.invoke(f"Add an unexpected twist to thsi story premise: {state['improved_story']}")
  return {"final_story":msg.content}

#Build the graph
builder=StateGraph(State)
builder.add_node("generate",generate_story)
builder.add_node("improve", improved_story)
builder.add_node("polish", polish_story)

#Define the edges
builder.add_edge(START, "generate")
builder.add_conditional_edges("generate",check_conflict,{"Pass":"improve","Fail":"generate"})
builder.add_edge("improve", "polish")
builder.add_edge("polish", END)

graph = builder.compile()

#View
graph.get_graph().draw_mermaid_png(output_file_path="PromptChaining-graph.png")

print("\n=== Prompt Chaining ===")
input_state: State = {
    "topic": "a fictitious detective like Sherlock Holmes",
    "story": "",
    "improved_story": "",
    "final_story": "",
}
result = graph.invoke(input_state)
print(result["final_story"])
print("\n====Improved Story ===")
print(result["improved_story"])
print("\n====Polished Story ===")
