import operator
from pathlib import Path
from typing import Annotated, List, cast
from dotenv import load_dotenv
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

class Section(BaseModel):
  name:str=Field(description="Name for this section of the report")
  description:str=Field(description="Brief Overview of the main topics and concepts of the section")

class Sections(BaseModel):
  sections:List[Section]=Field(
    description="Sections of the report"
  )

# Augment the LLM with schema for structured output
planner=llm_openai.with_structured_output(Sections)

## Creating Workers dynamically in LangGraph
## Because Orcherstrator-worker workflows are common, LangGraph has the Send API to support this.
## It lets you dynamically create worker nodes and send each one a specific input. Each worker
## has its own state, and all worker outputs are written to a shared state key that is accessible
## to the orchestrator graph. This gives the orchestrator access to all worker output and allows it
## to synthesize them into a final output. Below we iterate over a list of sections and send each
## to a worker node.

# Graph state
class State(TypedDict):
  topic:str # Report Topic
  sections: list[Section] #List of report sections
  completed_sections: Annotated[
    list, operator.add
  ] # All workers write to this key in parallel
  final_report: str # Final Report

# Worker State
class WorkerState(TypedDict):
  section: Section
  completed_sections: Annotated[list, operator.add]

# Nodes
def orchestrator(state: State):
  """Orchestrator that generates a plan for the report"""

  # Generate queries
  report_sections = cast(
    Sections,
    planner.invoke(
      [
        SystemMessage(content="Generate a plan for the report."),
        HumanMessage(content=f"Here is the report topic: {state['topic']}"),
      ]
    ),
  )

  print("Report Sections:",report_sections)

  return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
  """Worker wrties a section of the report"""

  # Generate Section
  section = llm_openai.invoke(
    [
      SystemMessage(
        content="Write a report section following the provided name and description. Include no preamble for each section"
      ),
      HumanMessage(
        content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
      )
    ]
  )

  # Write the updated section to completed sections (markdown heading per section)
  markdown_section = f"## {state['section'].name}\n\n{section.content}"
  return {"completed_sections": [markdown_section]}

# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: State):
  """Assign a worker to each section in the plan"""

  # Kick off section writing in parallel via Send() API
  return [Send("llm_call", {"section": s}) for s in state["sections"]]

def synthesizer(state: State):
  """Synthesize full report from sections"""

  # List of completed sections
  completed_sections = state["completed_sections"]

  # Combine sections into a markdown report
  completed_report_sections = f"# {state['topic']}\n\n" + "\n\n".join(completed_sections)

  return {"final_report": completed_report_sections}

## Build WorkFlow
orchestrator_worker_builder = StateGraph(State)

# Add the nodes
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
  "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

#Compile the Workflow
orchestrator_worker = orchestrator_worker_builder.compile()

# View the Workflow
orchestrator_worker.get_graph().draw_mermaid_png(output_file_path="OrchestratorWorker-graph.png")

# Invoke
input_state: State = {
    "topic": "Create a report on Agentic AI RAGs",
    "sections": [],
    "completed_sections": [],
    "final_report": "",
}
result = orchestrator_worker.invoke(input_state)

# Save final_report as markdown file
report_path = Path("Agentic-AI-RAGs-report.md")
report_path.write_text(result["final_report"], encoding="utf-8")
print(f"\nMarkdown report saved to: {report_path.resolve()}")
print(result["final_report"][:500], "...\n")

