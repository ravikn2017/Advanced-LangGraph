# # Quick illustration of the two approaches

# ### Deterministic Guardrails
# - Rule-based: regex, keyword matching, explicit checks
# - ✅ Fast, predictable, cost-effective
# - ❌ May miss nuanced violations

# ### Model-Based Guardrails
# - Uses LLMs/classifiers for semantic understanding
# - ✅ Catches subtle/nuanced issues
# - ❌ Slower and more expensive

import re

from dotenv import load_dotenv

load_dotenv()

# --- Deterministic approach ---
def deterministic_guardrail(text: str) -> bool:
    """Returns True if content is blocked."""
    banned_keywords = ["hack", "exploit", "malware", "bomb"]
    return any(kw in text.lower() for kw in banned_keywords)

test_inputs = [
    "How do I hack into a database?",
    "What is the capital of France?",
    "Explain how malware spreads",
]

print("=== Deterministic Guardrail Demo ===")
for inp in test_inputs:
    blocked = deterministic_guardrail(inp)
    status = "🚫 BLOCKED" if blocked else "✅ ALLOWED"
    print(f"{status}: {inp}")

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# --- Model-based approach ---
def model_based_guardrail(text: str) -> str:
    """Uses an LLM to evaluate content safety. Returns SAFE or UNSAFE."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Is the following user input safe to process? 
Reply with only 'SAFE' or 'UNSAFE'.

Input: {text}"""
    result = model.invoke([HumanMessage(content=prompt)])
    return str(result.content).strip()

print("=== Model-Based Guardrail Demo ===")
for inp in test_inputs:
    verdict = model_based_guardrail(inp)
    status = "🚫 UNSAFE" if "UNSAFE" in verdict else "✅ SAFE"
    print(f"{status}: {inp}")