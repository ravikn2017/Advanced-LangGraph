import operator
from pathlib import Path
from typing import Annotated, List, Literal, cast
from dotenv import load_dotenv
from langchain_core.callbacks import LLMManagerMixin
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

load_dotenv()

llm_openai = ChatOpenAI(model="gpt-4o")
# result=llm_openai.invoke("Hello")
# print(result)

# Graph State
class State(TypedDict):
  joke: str
  topic: str
  feedback: str
  funny_or_not: str

# Schema for structured output to use in evaluation
class Feedback(BaseModel):
  grade: Literal["funny", "not funny"] = Field(
    description="Decide if the joke is funny or not",
  )
  feedback: str = Field(
    description="If the joke is not funny, provide feedback on how to improve it"
  )

# Augment the LLM with schema for structured output
evaluator = llm_openai.with_structured_output(Feedback)

# Nodes
def llm_call_generator(state: State):
  """LLM generates a joke"""

  if state.get("feedback"):
    msg = llm_openai.invoke(
      f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
    )
  else:
    msg = llm_openai.invoke(f"Write a joke about {state['topic']}")
  return {"joke": msg.content}

def llm_call_evaluator(state: State):
  """LLM evaluates the joke"""

  grade = cast(Feedback, evaluator.invoke(f"Grade the joke {state['joke']}"))
  return {"funny_or_not": grade.grade, "feedback": grade.feedback}

# Conditional edge function to route back to joke generator or end-based upon feedback from the evaluator
def route_joke(state: State):
  """Route back to joke generator or end based upon feedback from the evaluator"""

  if state["funny_or_not"] == "funny":
    return "Accepted"
  elif state["funny_or_not"] == "not funny":
    return "Rejected + Feedback" 
  
# Build Workflow
optimizer_builder = StateGraph(State)

# Add the nodes
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# Add edges to connect nodes
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
  "llm_call_evaluator",
  route_joke,
  {# Name returned by route_joke : Name of next node to exit
    "Accepted": END,
    "Rejected + Feedback": "llm_call_generator",
  }

)

#Complile the Workflow
optimizer_workflow = optimizer_builder.compile()

#Show the Workflow
optimizer_workflow.get_graph().draw_mermaid_png(output_file_path="EvaluatorOptimizer-graph.png")

# Invoke
input_state: State = {
    "topic": "My daugther is in 12th grade and is not sincerely studying to get into good engineering college, always distracted by phone",
    "joke": "",
    "feedback": "",
    "funny_or_not": "",
}
result = optimizer_workflow.invoke(input_state)
print(result["joke"])