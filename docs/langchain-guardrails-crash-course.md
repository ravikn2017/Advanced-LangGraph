# Guardrails with LangChain — Crash Course

**By Krish Naik | KRISHAI Technologies**

This notebook covers everything you need to know about implementing **Guardrails** in LangChain agents using the middleware system.

### Topics Covered

1. What are Guardrails & Why do they matter?
2. Two approaches: Deterministic vs Model-based
3. Built-in: PII Detection Middleware
4. Built-in: Human-in-the-Loop Middleware
5. Custom: Before-Agent Guardrail (input filtering)
6. Custom: After-Agent Guardrail (output safety)
7. Layered / Combined Guardrails
8. Real-World Use Case: Healthcare Chatbot

---

> **Docs Reference:** https://docs.langchain.com/oss/python/langchain/guardrails

---

## Section 1: What are Guardrails?

Guardrails help you build **safe, compliant AI applications** by validating and filtering content at key points in your agent's execution.

They are implemented as **middleware** that intercepts execution:

- **Before** the agent starts (input guardrails)
- **After** it completes (output guardrails)
- **Around** model and tool calls

### Common Use Cases

| Use Case                  | Example                                   |
| ------------------------- | ----------------------------------------- |
| PII leakage prevention    | Redact emails/credit cards before logging |
| Prompt injection blocking | Detect adversarial inputs                 |
| Harmful content filtering | Block dangerous requests                  |
| Business rule enforcement | Require approval for financial ops        |
| Output quality validation | Ensure response meets safety standards    |

---

## Section 2: Two Approaches to Guardrails

### Deterministic Guardrails

- Rule-based: regex, keyword matching, explicit checks
- Fast, predictable, cost-effective
- May miss nuanced violations

### Model-Based Guardrails

- Uses LLMs/classifiers for semantic understanding
- Catches subtle/nuanced issues
- Slower and more expensive

---

## Section 3: Built-in Guardrail — PII Detection Middleware

LangChain provides built-in `PIIMiddleware` for detecting and handling **Personally Identifiable Information (PII)**.

### Supported PII Types

| Type          | Example                 |
| ------------- | ----------------------- |
| `email`       | user@example.com        |
| `credit_card` | 5105-1051-0510-5100     |
| `ip`          | 192.168.1.1             |
| `mac_address` | 00:1A:2B:3C:4D:5E       |
| `url`         | https://secret-site.com |

### Strategies

| Strategy | Result                |
| -------- | --------------------- |
| `redact` | `[REDACTED_EMAIL]`    |
| `mask`   | `****-****-****-1234` |
| `hash`   | `a8f5f167...`         |
| `block`  | Raises an exception   |

---

## Section 4: Built-in Guardrail — Human-in-the-Loop Middleware

Pauses agent execution before sensitive operations and waits for human approval.

**Best for:**

- Financial transactions
- Sending emails to external parties
- Deleting production data
- Any operation with significant business impact

**Key requirement:** A `checkpointer` for state persistence across interrupts.

---

## Section 5: Custom Guardrail — Before-Agent Hook (Input Filter)

Use `before_agent()` to validate or block requests **before any LLM processing begins**.

**Best for:**

- Keyword/content filtering
- Authentication checks
- Rate limiting
- Blocking specific categories of requests

---

## Section 6: Custom Guardrail — After-Agent Hook (Output Safety)

Use `after_agent()` to validate the final agent response **before the user sees it**.

**Best for:**

- Model-based safety evaluation of outputs
- Compliance scanning (e.g. legal, medical, financial disclaimers)
- Quality validation
- Removing sensitive info that slipped through

---

## Section 7: Layered / Combined Guardrails

Stack multiple guardrails in the `middleware=[]` array. They execute **in order**, building layered protection.

```
User Input
    ↓
[Layer 1] ContentFilterMiddleware    ← Deterministic input filter
    ↓
[Layer 2] PIIMiddleware (input)      ← PII redaction on input
    ↓
[Layer 3] HumanInTheLoopMiddleware   ← Approval for sensitive tools
    ↓
[Layer 4] PIIMiddleware (output)     ← PII redaction on output
    ↓
[Layer 5] SafetyGuardrailMiddleware  ← Model-based output safety
    ↓
User Response
```

---

## Section 8: Real-World Use Case — Healthcare Chatbot

A healthcare chatbot that:

1. **Blocks** off-topic or harmful requests
2. **Redacts** patient PII (emails, credit card numbers)
3. **Requires human approval** before booking appointments
4. **Validates** that outputs are medically appropriate

---

## Summary

| Guardrail Type    | Hook           | When it Runs           | Best For                  |
| ----------------- | -------------- | ---------------------- | ------------------------- |
| PII Middleware    | Input/Output   | Around model calls     | Data privacy, compliance  |
| Human-in-the-Loop | Tool level     | Before sensitive tools | High-stakes decisions     |
| Content Filter    | `before_agent` | Start of invocation    | Blocking bad inputs early |
| Safety Validator  | `after_agent`  | End of invocation      | Output quality/safety     |
| Custom Logic      | Any hook       | Anywhere               | Any business rule         |

### Key Takeaways

1. **Guardrails = Middleware** — implement them via the `middleware=[]` parameter in `create_agent()`
2. **Layer your guardrails** — defense in depth is best practice
3. **Deterministic first, model-based second** — use cheap rule-based checks early to avoid expensive LLM calls
4. **Human-in-the-Loop requires a checkpointer** — use `InMemorySaver` for dev, persistent store for production
5. **Custom middleware** gives you full control via `before_agent()` and `after_agent()` hooks

---

### Additional Resources

- [LangChain Guardrails Docs](https://docs.langchain.com/oss/python/langchain/guardrails)
- [Middleware Docs](https://docs.langchain.com/oss/python/langchain/middleware/overview)
- [Human-in-the-Loop Docs](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- [LangSmith for Observability](https://docs.langchain.com/oss/python/langchain/observability)

---
