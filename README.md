# LLM-Powered Research Agent

An end-to-end demonstration of LLM tool calling and structured output using the **Anthropic Claude** API.

---

## What the Project Does

The agent accepts natural-language research questions, autonomously decides which tools to call (and in what order), executes them, and returns either a direct answer or a fully-validated structured JSON report.

```
User query → Claude → [tool call(s)] → results → Claude → final answer / structured report
```

---

## Project Structure

```
Assignment-4/
├── agent.py            # Core agent: tool loop, structured output, temp experiments
├── langchain_agent.py  # Part 4 (optional): LangChain re-implementation
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python agent.py
```

---

## Part 1 — LLM API Setup

**Provider:** Anthropic (Claude Sonnet 4.6)

**Error handling & retries** via `tenacity`:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
)
def call_claude(...): ...
```

**Decoding parameter experiments** — `experiment_temperatures()` runs the same prompt at temperatures `0.0`, `0.5`, and `1.0`:

| Temperature | Behavior |
|---|---|
| 0.0 | Deterministic, factual, repetitive across runs |
| 0.5 | Balanced — coherent but with some variation |
| 1.0 | Creative, diverse, occasionally less precise |

*Observation:* For structured JSON output we use `temperature=0.2`; for creative tasks `0.7–1.0` is preferable.

---

## Part 2 — Structured Output

Schemas defined with **Pydantic v2**:

```python
class ResearchEntity(BaseModel):
    name: str
    entity_type: str          # person | place | organization | concept
    relevance_score: float    # 0.0 – 1.0
    description: str

class ResearchReport(BaseModel):
    title: str
    summary: str
    key_entities: list[ResearchEntity]
    tools_used: list[str]
    confidence: float         # 0.0 – 1.0
    timestamp: str
    raw_findings: list[str]
```

**Enforcement strategy:** The system prompt instructs Claude to respond with *only* valid JSON matching the schema. The output is parsed with `json.loads()` then validated with Pydantic (`ResearchReport(**data)`). A fallback minimal report is generated if parsing fails.

### Example structured output

```json
{
  "title": "Machine Learning and Python: Research Report",
  "summary": "Python is the dominant language for machine learning due to libraries like scikit-learn, TensorFlow, and PyTorch. ML enables systems to learn patterns from data without explicit programming.",
  "key_entities": [
    {"name": "Python", "entity_type": "concept", "relevance_score": 0.95, "description": "Primary programming language for ML"},
    {"name": "scikit-learn", "entity_type": "organization", "relevance_score": 0.85, "description": "Python ML library"}
  ],
  "tools_used": ["web_search"],
  "confidence": 0.82,
  "timestamp": "2025-10-15T14:22:01.123456",
  "raw_findings": ["Python created 1991 by Guido van Rossum", "ML key algorithms: neural networks, SVMs, decision trees"]
}
```

---

## Part 3 — Tool Calling

### Tools defined

| Tool | Description | Inputs |
|---|---|---|
| `calculator` | Safe math expression evaluator | `expression: str` |
| `web_search` | Simulated web search (mock DB) | `query: str` |
| `get_weather` | Current weather for a city (mock) | `city: str` |

### Tool-calling loop (`run_tool_loop`)

1. Send user message + tool schemas to Claude
2. If `stop_reason == "tool_use"` → extract `tool_use` blocks, execute each tool, append `tool_result` messages
3. Loop until `stop_reason == "end_turn"` or max iterations reached

### Example interactions

**Multi-tool query:**
```
Input:  "What's the weather in Tokyo, and what is sqrt(144) + 2^8?"
Tools:  get_weather("Tokyo"), calculator("sqrt(144) + 2**8")
Output: "Tokyo: 8°C, Clear. sqrt(144) + 2^8 = 12 + 256 = 268"
```

**No tools needed:**
```
Input:  "What is the capital of France?"
Tools:  (none)
Output: "The capital of France is Paris."
```

**Sequential tool use (research report):**
```
Input:  "Research machine learning and Python"
Tools:  web_search("machine learning") → web_search("Python programming") → structured JSON report
```

---

## Part 4 — LangChain Integration (`langchain_agent.py`)

Uses `create_tool_calling_agent` + `AgentExecutor` with `ConversationBufferMemory` for multi-turn dialogue.

```python
agent = create_langchain_agent(temperature=0.4)
result = agent.invoke({"input": "What is 15 * 24 + sqrt(625)?"})
# Memory persists: next question can reference "what I asked earlier"
```

**LangChain vs scratch comparison:**

| Aspect | From Scratch | LangChain |
|---|---|---|
| Control | Full | Framework-managed |
| Boilerplate | More | Less |
| Memory | Manual | `ConversationBufferMemory` |
| Debugging | Easier | `verbose=True` helps |
| Flexibility | Maximum | Plugin ecosystem |

---

## Design Decisions

1. **Temperature 0.2 for structured output** — minimises hallucinated field names or invalid JSON
2. **Pydantic for validation** — catches type errors and missing fields the JSON parser would miss
3. **Mock tools** — avoids API key requirements for search/weather; swap in real APIs (Brave Search, OpenWeatherMap) by replacing the function bodies only
4. **Max iterations guard** — prevents infinite tool-calling loops if the model gets confused
5. **Tenacity for retries** — exponential backoff is standard practice for LLM API calls
