import json
import math
import time
import random
from typing import Any
from datetime import datetime

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
load_dotenv()

# Structured Output Schemas


class ResearchEntity(BaseModel):
    """A named entity extracted from research text."""
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Type: person, place, organization, concept")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance 0-1")
    description: str = Field(description="Brief description")


class ResearchReport(BaseModel):
    """Structured research report produced by the agent."""
    title: str = Field(description="Report title")
    summary: str = Field(description="Executive summary (2-3 sentences)")
    key_entities: list[ResearchEntity] = Field(description="Key entities found")
    tools_used: list[str] = Field(description="Names of tools invoked")
    confidence: float = Field(ge=0.0, le=1.0, description="Agent confidence in findings")
    timestamp: str = Field(description="ISO timestamp of report generation")
    raw_findings: list[str] = Field(description="Bullet-point findings")



#Tool Definitions & Implementations


#  Tool 1: Calculator 
def calculator(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt, log, sin, cos, tan, pi, e
    """
    allowed_names = {
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e, "abs": abs, "round": round,
        "pow": math.pow, "exp": math.exp,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return {"result": result, "expression": expression, "status": "success"}
    except Exception as exc:
        return {"result": None, "expression": expression, "status": "error", "error": str(exc)}


#  Tool 2: Simulated Web Search 
MOCK_SEARCH_DB = {
    "python": "Python is a high-level, interpreted programming language known for readability. Created by Guido van Rossum in 1991. Used widely in data science, web development, and AI.",
    "machine learning": "Machine learning is a subset of AI that enables systems to learn from data. Key algorithms: linear regression, neural networks, decision trees, SVMs.",
    "anthropic": "Anthropic is an AI safety company founded in 2021. They created Claude, a family of large language models focused on safety and helpfulness.",
    "climate change": "Climate change refers to long-term shifts in global temperatures and weather patterns. Since the 1800s, human activities have been the main driver via greenhouse gas emissions.",
    "quantum computing": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information. Promises exponential speedups for certain problems.",
    "default": "No specific results found. General knowledge: The topic you searched may require more specific terms.",
}

def web_search(query: str) -> dict:
    """
    Simulate a web search (returns mock results for demo purposes).
    In production, replace with a real search API like Brave or SerpAPI.
    """
    query_lower = query.lower()
    result = MOCK_SEARCH_DB.get("default")
    for key, value in MOCK_SEARCH_DB.items():
        if key in query_lower:
            result = value
            break
    # Simulate slight network delay
    time.sleep(0.1)
    return {
        "query": query,
        "results": result,
        "source": "MockSearchEngine v1.0",
        "timestamp": datetime.now().isoformat(),
    }


#  Tool 3: Weather Lookup 
MOCK_WEATHER = {
    "london": {"temp_c": 12, "condition": "Cloudy", "humidity": 78, "wind_kph": 20},
    "new york": {"temp_c": 5, "condition": "Partly Cloudy", "humidity": 55, "wind_kph": 15},
    "tokyo": {"temp_c": 8, "condition": "Clear", "humidity": 60, "wind_kph": 10},
    "sydney": {"temp_c": 24, "condition": "Sunny", "humidity": 65, "wind_kph": 12},
    "paris": {"temp_c": 9, "condition": "Rainy", "humidity": 85, "wind_kph": 18},
}

def get_weather(city: str) -> dict:
    """Return current weather conditions for a city (mock data)."""
    city_lower = city.lower()
    data = MOCK_WEATHER.get(city_lower)
    if not data:
        # Random plausible weather for unknown cities
        data = {
            "temp_c": random.randint(-5, 35),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Windy"]),
            "humidity": random.randint(30, 90),
            "wind_kph": random.randint(5, 40),
        }
    return {"city": city, **data, "temp_f": round(data["temp_c"] * 9 / 5 + 32, 1)}



# Tool Registry: maps tool names → functions


TOOL_REGISTRY: dict[str, callable] = {
    "calculator": calculator,
    "web_search": web_search,
    "get_weather": get_weather,
}

# Claude tool schemas (passed to API)
TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Evaluate mathematical expressions. Supports arithmetic, sqrt, log, trig functions, pi, e.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression, e.g. '2 ** 10' or 'sqrt(144) + log(100)'"
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for information on a topic. Returns a text summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city. Returns temperature, condition, humidity, wind.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'London', 'Tokyo'"
                }
            },
            "required": ["city"],
        },
    },
]



# Part 1: API Setup with Retry + Rate Limiting


client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
    reraise=True,
)
def call_claude(
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    system: str = "",
) -> anthropic.types.Message:
    """
    Call Claude API with retry logic for rate limits and connection errors.
    
    Decoding parameters:
    - temperature: Controls randomness (0=deterministic, 1=creative)
    - max_tokens: Cap on response length
    Note: Claude API doesn't expose top_k/top_p directly in all SDKs,
          but temperature is the primary lever for output diversity.
    """
    kwargs: dict[str, Any] = {
        "model": "claude-sonnet-4-6",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools

    return client.messages.create(**kwargs)



# Tool Calling Loop


def run_tool_loop(
    user_message: str,
    system_prompt: str = "",
    temperature: float = 0.3,
    verbose: bool = True,
) -> tuple[str, list[str]]:
    """
    Full agentic tool-calling loop:
    1. Send user message to Claude
    2. Claude decides to call tool(s) or respond directly
    3. Execute tool(s), feed results back
    4. Repeat until Claude gives a final text response
    
    Returns (final_answer, list_of_tools_used)
    """
    messages = [{"role": "user", "content": user_message}]
    tools_used = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"USER: {user_message}")
        print(f"{'='*60}")

    iteration = 0
    max_iterations = 10  # safety guard against infinite loops

    while iteration < max_iterations:
        iteration += 1
        response = call_claude(
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=temperature,
            system=system_prompt,
        )

        if verbose:
            print(f"\n[Iteration {iteration}] Stop reason: {response.stop_reason}")

        # Check if we're done (no more tool calls)
        if response.stop_reason == "end_turn":
            # Extract text from response
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            if verbose:
                print(f"\nASSISTANT: {final_text[:300]}{'...' if len(final_text) > 300 else ''}")
            return final_text, tools_used

        # Handle tool use
        if response.stop_reason == "tool_use":
            # Add assistant's response (including tool calls) to messages
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tools_used.append(tool_name)

                    if verbose:
                        print(f"\n  → Calling tool: {tool_name}({json.dumps(tool_input)})")

                    # Execute the tool
                    if tool_name in TOOL_REGISTRY:
                        result = TOOL_REGISTRY[tool_name](**tool_input)
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}

                    if verbose:
                        print(f"  ← Result: {json.dumps(result)[:200]}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            # Feed tool results back
            messages.append({"role": "user", "content": tool_results})

    return "Max iterations reached without final answer.", tools_used



# Part 2: Structured Output Generation


REPORT_SYSTEM_PROMPT = """You are a research analyst. After gathering information using tools,
produce a structured JSON report. Your response must be ONLY valid JSON matching this exact schema:

{
  "title": "string",
  "summary": "string (2-3 sentences)",
  "key_entities": [
    {
      "name": "string",
      "entity_type": "person|place|organization|concept",
      "relevance_score": float between 0 and 1,
      "description": "string"
    }
  ],
  "tools_used": ["list", "of", "tool", "names"],
  "confidence": float between 0 and 1,
  "timestamp": "ISO datetime string",
  "raw_findings": ["bullet finding 1", "bullet finding 2", ...]
}

Do not include any text outside the JSON object."""


def generate_research_report(topic: str, verbose: bool = True) -> ResearchReport:
    """
    Full pipeline: research a topic using tools, then produce a structured report.
    
    Uses lower temperature (0.2) for structured output to ensure reliable JSON parsing.
    """
    # Step 1: Research phase - use tools to gather info
    research_prompt = f"""Research the following topic thoroughly using available tools:

Topic: {topic}

Use web_search to find relevant information. If the topic involves numbers or calculations,
use the calculator tool. If weather context is relevant, use get_weather.
After gathering information, write a comprehensive research report as structured JSON."""

    raw_answer, tools_used = run_tool_loop(
        user_message=research_prompt,
        system_prompt=REPORT_SYSTEM_PROMPT,
        temperature=0.2,  # Low temp for reliable structured output
        verbose=verbose,
    )

    # Step 2: Parse and validate with Pydantic
    try:
        # Strip any markdown code fences if present
        cleaned = raw_answer.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        data = json.loads(cleaned)
        # Inject tools_used if not already present
        if not data.get("tools_used"):
            data["tools_used"] = list(set(tools_used))
        if not data.get("timestamp"):
            data["timestamp"] = datetime.now().isoformat()

        report = ResearchReport(**data)
        return report

    except (json.JSONDecodeError, ValidationError) as exc:
        print(f"\n[WARNING] Structured parsing failed: {exc}")
        print("[WARNING] Falling back to minimal report...")
        # Fallback: create a minimal valid report
        return ResearchReport(
            title=f"Research: {topic}",
            summary=raw_answer[:300] if raw_answer else "Research completed.",
            key_entities=[],
            tools_used=list(set(tools_used)),
            confidence=0.5,
            timestamp=datetime.now().isoformat(),
            raw_findings=[raw_answer[:500]] if raw_answer else [],
        )



# Part 1: Decoding Parameter Experiments


def experiment_temperatures(prompt: str = "Write a creative opening line for a sci-fi novel."):
    """
    Demonstrate how temperature affects LLM output diversity and creativity.
    Runs same prompt at temperatures 0.0, 0.5, 1.0 and compares outputs.
    """
    print("\n" + "="*60)
    print("TEMPERATURE EXPERIMENT")
    print("="*60)
    print(f"Prompt: {prompt}\n")

    for temp in [0.0, 0.5, 1.0]:
        messages = [{"role": "user", "content": prompt}]
        response = call_claude(messages=messages, temperature=temp, max_tokens=150)
        text = response.content[0].text if response.content else ""
        print(f"Temperature {temp:.1f}: {text.strip()}")
        print("-" * 40)



# Main Demo


if __name__ == "__main__":
    print("LLM Research Agent — Full Demo")
    print("="*60)

    #  Demo 1: Simple tool calling 
    print("\n### DEMO 1: Multi-tool query ###")
    answer, tools = run_tool_loop(
        "What's the weather in Tokyo, and what is sqrt(144) + 2^8?",
        temperature=0.3,
    )
    print(f"\nTools used: {tools}")

    #  Demo 2: Tool calling that needs no tools 
    print("\n### DEMO 2: No tools needed ###")
    answer2, tools2 = run_tool_loop(
        "What is the capital of France?",
        temperature=0.3,
    )
    print(f"\nTools used: {tools2} (empty = no tools needed)")

    #  Demo 3: Structured research report 
    print("\n### DEMO 3: Structured Research Report ###")
    report = generate_research_report("machine learning and Python", verbose=True)
    print("\n" + "="*60)
    print("PARSED STRUCTURED REPORT:")
    print("="*60)
    print(f"Title:      {report.title}")
    print(f"Summary:    {report.summary}")
    print(f"Confidence: {report.confidence:.0%}")
    print(f"Tools used: {report.tools_used}")
    print(f"Entities:   {[e.name for e in report.key_entities]}")
    print(f"Findings:   {len(report.raw_findings)} items")

    #  Demo 4: Temperature experiment 
    experiment_temperatures()
