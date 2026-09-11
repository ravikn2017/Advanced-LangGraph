# Built-in Guardrail — PII Detection Middleware

# LangChain provides built-in `PIIMiddleware` for detecting and handling **Personally Identifiable Information (PII)**.

# ### Supported PII Types:
# | Type | Example |
# |---|---|
# | `email` | user@example.com |
# | `credit_card` | 5105-1051-0510-5100 |
# | `ip` | 192.168.1.1 |
# | `mac_address` | 00:1A:2B:3C:4D:5E |
# | `url` | https://secret-site.com |

# ### Strategies:
# | Strategy | Result |
# |---|---|
# | `redact` | `[REDACTED_EMAIL]` |
# | `mask` | `****-****-****-1234` |
# | `hash` | `a8f5f167...` |
# | `block` | Raises an exception |

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from dotenv import load_dotenv

load_dotenv()

# Define a simple dummy tool
@tool
def customer_lookup(query: str) -> str:
    """Look up customer information."""
    return f"Customer record found for query: {query}"

# Create agent with PII Middleware
agent = create_agent(
    model="gpt-4o",
    tools=[customer_lookup],
    middleware=[
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)

print("Agent with PII middleware created successfully!")

# Test PII Redaction
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "My email is john.doe@example.com and my card is 5105-1051-0510-5100. Can you help me?"
    }]
})

print("=== Agent Response ===")
print(result["messages"][-1].content)

# Test API Key Blocking
try:
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Here is my key: sk-abcdefghijklmnopqrstuvwxyz123456"
        }]
    })
    
except Exception as e:
    print(f"🚫 Blocked as expected: {e}")