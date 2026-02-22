from datetime import datetime
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import math



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



# LangChain Agent Setup


def create_langchain_agent(temperature: float = 0.3):
    """
    Build a LangChain agent with:
    - Claude as the LLM backbone
    - 3 tools: calculator, web_search, get_weather
    - Conversation memory for multi-turn dialogue
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=temperature,
        max_tokens=2048,
    )

    tools = [lc_calculator, lc_web_search, lc_get_weather]

    # Prompt with system message and chat history placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful research assistant with access to tools.
Use tools when needed to provide accurate, up-to-date information.
Be concise but thorough. If a question doesn't need tools, answer directly."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create tool-calling agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Wrap with executor (handles the tool loop)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,       # Shows tool calls in console
        max_iterations=8,
        handle_parsing_errors=True,
    )
    return executor



# Conversational Demo


def run_conversation():
    """
    Run a multi-turn conversation demonstrating memory persistence.
    The agent remembers earlier turns when answering later questions.
    """
    print("\n" + "="*60)
    print("LANGCHAIN CONVERSATIONAL AGENT DEMO")
    print("="*60)

    agent = create_langchain_agent(temperature=0.4)

    turns = [
        "What is 15 * 24 + sqrt(625)?",
        "Now tell me about machine learning.",
        "What's the weather in Paris right now?",
        "Based on what I asked you so far, what topics have we discussed?",  # Memory test
    ]

    for i, question in enumerate(turns, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Human: {question}")
        result = agent.invoke({"input": question})
        print(f"Agent: {result['output']}")


if __name__ == "__main__":
    run_conversation()
