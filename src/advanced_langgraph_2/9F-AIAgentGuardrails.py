# ---
# ## 🧱 Section 7: Layered / Combined Guardrails

# Stack multiple guardrails in the `middleware=[]` array. They execute **in order**, building layered protection.

# ```
# User Input
#     ↓
# [Layer 1] ContentFilterMiddleware    ← Deterministic input filter
#     ↓
# [Layer 2] PIIMiddleware (input)      ← PII redaction on input
#     ↓
# [Layer 3] HumanInTheLoopMiddleware   ← Approval for sensitive tools
#     ↓
# [Layer 4] PIIMiddleware (output)     ← PII redaction on output
#     ↓
# [Layer 5] SafetyGuardrailMiddleware  ← Model-based output safety
#     ↓
# User Response
# ```

import importlib.util
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool

from dotenv import load_dotenv

load_dotenv()


def _import_sibling_module(module_name: str, file_name: str):
    """Import middleware classes from sibling guardrail scripts."""
    path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_guardrails_d = _import_sibling_module("guardrails_d", "9D-AIAgentGuardrails.py")
_guardrails_e = _import_sibling_module("guardrails_e", "9E-AIAgentGuardrails.py")
ContentFilterMiddleware = _guardrails_d.ContentFilterMiddleware
SafetyGuardrailMiddleware = _guardrails_e.SafetyGuardrailMiddleware

@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Search results: {query}"

@tool
def send_email_tool(to: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

# Full layered guardrail stack
production_agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, send_email_tool],
    middleware=[
        # Layer 1: Deterministic input filter (before agent)
        ContentFilterMiddleware(banned_keywords=["hack", "exploit", "malware"]),

        # Layer 2: PII redaction on input
       
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),

        # Layer 3: Human approval for sensitive tools
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email_tool": True, "search_tool": False}
        ),

        # Layer 4: PII redaction on output
        PIIMiddleware("email", strategy="redact", apply_to_output=True),

        # Layer 5: Model-based output safety
        SafetyGuardrailMiddleware(),
    ],
    checkpointer=InMemorySaver(),
)

print("🏭 Production-grade agent with 5-layer guardrails created!")