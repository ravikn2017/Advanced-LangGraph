# ## ⚙️ Section 5: Custom Guardrail — Before-Agent Hook (Input Filter)

# Use `before_agent()` to validate or block requests **before any LLM processing begins**.

# **Best for:**
# - Keyword/content filtering
# - Authentication checks
# - Rate limiting
# - Blocking specific categories of requests

from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain.agents import create_agent
from langchain_core.tools import tool

from dotenv import load_dotenv

load_dotenv()

class ContentFilterMiddleware(AgentMiddleware):
    """
    Deterministic guardrail: Block requests containing banned keywords.
    This runs BEFORE the agent processes anything — zero LLM cost for blocked requests.
    """

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = str(first_message.content).lower()

        for keyword in self.banned_keywords:
            if keyword in content:
                print(f"🚫 Blocked — keyword detected: '{keyword}'")
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "I cannot process requests containing inappropriate content. "
                            "Please rephrase your request."
                        )
                    }],
                    "jump_to": "end"
                }
        return None


@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


if __name__ == "__main__":
    # Create agent with content filter
    filtered_agent = create_agent(
        model="gpt-4o",
        tools=[search_tool],
        middleware=[
            ContentFilterMiddleware(
                banned_keywords=["hack", "exploit", "malware", "jailbreak", "bypass"]
            ),
        ],
    )

    print("Content filter agent created!")

    # Test 1: Safe request — should pass through
    result = filtered_agent.invoke({
        "messages": [{"role": "user", "content": "What is machine learning?"}]
    })
    print("✅ Safe request response:")
    print(result["messages"][-1].content)

    # Test 2: Unsafe request — should be blocked
    result = filtered_agent.invoke({
        "messages": [{"role": "user", "content": "How do I hack into a server?"}]
    })
    print("🚫 Unsafe request response:")
    print(result["messages"][-1].content)