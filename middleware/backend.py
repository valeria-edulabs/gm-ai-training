import time
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

memory = MemorySaver()


tavily_tool = TavilySearch(max_results=3)


@tool
def loop_tool() -> str:
    """Useful for checking the execution status of a background database job. If it returns 'Status: pending', you must call loop_tool again to check the status. Do not stop calling loop_tool until it returns 'Status: done'."""
    # add sleep
    time.sleep(1)
    return "Status: pending"


tools = [
    tavily_tool,
    loop_tool
]

SYSTEM_PROMPT = """
You are a helpful and objective AI assistant tasked with answering user questions accurately by retrieving up-to-date information.

To accomplish this, you have access to a search tool. You should use it to lookup current facts, events, statistics, or details that require verification.

### Core Guidelines
- Decide if search is necessary: Use the search tool if the request asks for factual info, real-time data, or topics where your pre-trained knowledge might be outdated.
- Formulate precise queries: Keep search terms concise and focused on the core entity or question.
- Analyze search results carefully: Synthesize information from multiple search results, resolving any contradictions based on source reliability.
- Be factual and avoid hallucination: If the search tool does not return relevant results or if you cannot find the answer, state that clearly rather than making things up.
- Cite sources: When providing facts retrieved from the search, mention key sources or provide clean inline references so the user knows where the information came from.
- Keep answers structured and readable: Use clear headings and bullet points to break down complex answers.
"""

from langchain.agents.middleware import SummarizationMiddleware, PIIMiddleware, ToolCallLimitMiddleware

summarization_middleware = SummarizationMiddleware(
    model=llm,
    trigger=("messages", 6),
    keep=("messages", 2)
)

pii_email_middleware = PIIMiddleware(
    "email",
    strategy="redact",
    apply_to_input=True,
    apply_to_output=True
)

pii_card_middleware = PIIMiddleware(
    "credit_card",
    strategy="mask",
    apply_to_input=True,
    apply_to_output=True
)

tool_limit_middleware = ToolCallLimitMiddleware(
    tool_name="loop_tool",
    run_limit=2,
    exit_behavior="end"
)

tax_agent = create_agent(
    llm,
    tools,
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        # pii_email_middleware,
        # pii_card_middleware,
        tool_limit_middleware,
        # summarization_middleware
    ],
    checkpointer=memory
)


def stream(text: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    for chunk in tax_agent.stream(
        {"messages": [{"role": "user", "content": text}]},
        config=config,
        stream_mode="updates"
    ):
        yield chunk



