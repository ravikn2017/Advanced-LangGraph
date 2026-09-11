from typing import Literal, cast
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

load_dotenv()

llm_openai = ChatOpenAI(model="gpt-4o")
result=llm_openai.invoke("Hello")
print(result)

# Schema for structure output to use as routing logic
class Route(BaseModel):
  step:Literal["poem", "story", "joke"]=Field(description="The next step in the routing process")

## Augment the LLM with schema for structured output
router=llm_openai.with_structured_output(Route)

#Graph State
class State(TypedDict):
  input:str
  decision:str
  output:str

## Nodes
def llm_call_story(state:State):
  """Write a Story"""
  print("LLM Call Story is called")
  msg=llm_openai.invoke(state["input"])
  return {"story":msg.content}

def llm_call_joke(state:State):
  """Write a joke"""
  print("LLM Call Joke is called")

  result=llm_openai.invoke(state["input"])
  return {"output": result.content}

def llm_call_poem(state:State):
  """Write a poem"""
  print("LLM Call Poem is called")

  result=llm_openai.invoke(state["input"])
  return {"output": result.content}

def llm_call_router(state:State):
  """Route the input to the appropriate node"""

  decision = cast(
    Route,
    router.invoke(
      [
        SystemMessage(
          content="Route the input to story, joke or poem based on the users request"
        ),
        HumanMessage(content=state["input"]),
      ]
    ),
  )
  return {"decision": decision.step}

# Conditional Edge function to route to the appropriate node
def route_decision(state: State):
   # Return the node name you want to visit next
    if state["decision"] == "story":
      return "llm_call_story"
    elif state["decision"] == "joke":
      return "llm_call_joke"
    elif state["decision"] == "poem":
      return "llm_Call_poem"



#Build WorkFlow
router_builder = StateGraph(State)

#Add Nodes
router_builder.add_node("llm_call_story", llm_call_story)
router_builder.add_node("llm_call_joke", llm_call_joke)
router_builder.add_node("llm_call_poem", llm_call_poem)
router_builder.add_node("llm_call_router",llm_call_router)

#Add edges to connect nodes
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
   "llm_call_router",
  route_decision,
  { #Name returned by route_decision : Name of next node to visit
    "llm_call_story":"llm_call_story",
    "llm_call_joke":"llm_call_joke",
    "llm_call_poem":"llm_call_poem",
  },
)

router_builder.add_edge("llm_call_story", END)
router_builder.add_edge("llm_call_joke", END)
router_builder.add_edge("llm_call_poem", END)

#Compile WorkFlow
router_workflow = router_builder.compile()

# Show the workflow
router_workflow.get_graph().draw_mermaid_png(output_file_path="Router-graph.png")

# Get the output
input_state: State = {
    "input": "Write me a joke about Agentic AI System",
    "decision": "",
    "output": "",
}
state = router_workflow.invoke(input_state)
print(state["output"])