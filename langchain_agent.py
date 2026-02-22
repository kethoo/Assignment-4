"""
Part 4 (Optional): LangChain Agent Implementation
==================================================
Re-implements the research agent using LangChain for comparison.
Compatible with LangChain 1.x+
"""

import math
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
load_dotenv()



# Define LangChain Tools (using @tool decorator)


@tool
def lc_calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Supports: arithmetic, sqrt, log, sin, cos, tan, pi, e, abs, pow, exp.
    Example: 'sqrt(256) + log10(1000)'
    """
    allowed_names = {
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e, "abs": abs, "round": round,
        "pow": math.pow, "exp": math.exp,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return f"Result of '{expression}' = {result}"
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


@tool
def lc_web_search(query: str) -> str:
    """
    Search the web for information about a topic.
    Returns a summary of relevant results.
    """
    mock_db = {
        "python": "Python is a versatile, high-level programming language. Popular for data science, web dev, and AI.",
        "machine learning": "Machine learning (ML) is AI that learns from data. Key techniques: neural networks, SVMs, decision trees.",
        "anthropic": "Anthropic is an AI safety company that created Claude. Founded in 2021, focused on safe and beneficial AI.",
        "climate": "Climate change refers to long-term global temperature shifts, primarily caused by human greenhouse gas emissions.",
        "quantum": "Quantum computing uses quantum mechanics (superposition, entanglement) for computation. Promises breakthroughs in cryptography, drug discovery, and optimization.",
    }
    query_lower = query.lower()
    for key, val in mock_db.items():
        if key in query_lower:
            return val
    return f"Search results for '{query}': General information available. Topic may require more specific query terms."


@tool
def lc_get_weather(city: str) -> str:
    """
    Get current weather conditions for a city.
    Returns temperature (Celsius and Fahrenheit), condition, humidity, and wind speed.
    """
    import random
    mock_weather = {
        "london": (12, "Cloudy", 78, 20),
        "new york": (5, "Partly Cloudy", 55, 15),
        "tokyo": (8, "Clear", 60, 10),
        "sydney": (24, "Sunny", 65, 12),
        "paris": (9, "Rainy", 85, 18),
    }
    city_lower = city.lower()
    if city_lower in mock_weather:
        temp_c, condition, humidity, wind = mock_weather[city_lower]
    else:
        temp_c = random.randint(0, 30)
        condition = random.choice(["Sunny", "Cloudy", "Rainy"])
        humidity = random.randint(40, 80)
        wind = random.randint(5, 25)

    temp_f = round(temp_c * 9 / 5 + 32, 1)
    return f"{city}: {temp_c}°C ({temp_f}°F), {condition}, Humidity: {humidity}%, Wind: {wind} km/h"



# LangChain 1.x Agent Setup (manual loop)


TOOLS = [lc_calculator, lc_web_search, lc_get_weather]
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM = """You are a helpful research assistant with access to tools.
Use tools when needed to provide accurate, up-to-date information.
Be concise but thorough. If a question doesn't need tools, answer directly."""


def create_langchain_agent(temperature: float = 0.3):
    """
    Build a LangChain 1.x compatible agent.
    Uses bind_tools() to attach tools to the model.
    Returns (llm_with_tools, chat_history)
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=temperature,
        max_tokens=2048,
    )
    llm_with_tools = llm.bind_tools(TOOLS)
    return llm_with_tools


def run_agent_turn(llm_with_tools, history: list, user_input: str, verbose: bool = True) -> str:
    """
    Run one conversational turn with tool-calling loop.
    Appends messages to history for memory across turns.
    """
    history.append(HumanMessage(content=user_input))

    if verbose:
        print(f"\nHuman: {user_input}")

    iteration = 0
    while iteration < 8:
        iteration += 1
        messages = [SystemMessage(content=SYSTEM)] + history
        response = llm_with_tools.invoke(messages)

        history.append(response)

        # If no tool calls, we're done
        if not response.tool_calls:
            if verbose:
                print(f"Agent: {response.content}")
            return response.content

        # Execute tool calls
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            if verbose:
                print(f"  → Tool: {tool_name}({tool_args})")

            if tool_name in TOOL_MAP:
                result = TOOL_MAP[tool_name].invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"

            if verbose:
                print(f"  ← Result: {str(result)[:150]}")

            history.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return "Max iterations reached."



# Conversational Demo


def run_conversation():
    """
    Run a multi-turn conversation demonstrating memory persistence.
    The agent remembers earlier turns when answering later questions.
    """
    print("\n" + "="*60)
    print("LANGCHAIN CONVERSATIONAL AGENT DEMO")
    print("="*60)

    llm_with_tools = create_langchain_agent(temperature=0.4)
    history = []  # Shared history = memory

    turns = [
        "What is 15 * 24 + sqrt(625)?",
        "Now tell me about machine learning.",
        "What's the weather in Paris right now?",
        "Based on what I asked you so far, what topics have we discussed?",  # Memory test
    ]

    for i, question in enumerate(turns, 1):
        print(f"\n--- Turn {i} ---")
        run_agent_turn(llm_with_tools, history, question)


if __name__ == "__main__":
    run_conversation()