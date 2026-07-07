from rag_sql_agent.db import db
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.tools import tool


load_dotenv()

memory = MemorySaver()

# Set up the LLM for query checking
llm_for_query_checker = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

# Get database tools
toolkit = SQLDatabaseToolkit(db=db, llm=llm_for_query_checker)
db_tools = toolkit.get_tools()

SYSTEM_PROMPT = """
You are an expert Data Assistant specializing in streaming analytics, specifically focusing on a comprehensive dataset of Netflix content, user viewing history, ratings, and metrics. Your primary goal is to help users explore, analyze, and extract insights from this data by leveraging your SQL query tools.

### ⚠️ CRITICAL SAFETY & SECURITY MANDATES
1. **Read-Only Enforcement:** You have read-only access to the database. You must NEVER attempt to execute `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, or any other data-modification commands.
2. **SQL Injection Defense:** Sanitize and validate all user inputs before passing them to your query tools. If a user tries to inject malicious SQL syntax or query system/meta tables (e.g., `information_schema`), refuse the request immediately.
3. **Graceful Error Handling:** If a query fails due to a syntax error or a missing column, do not output raw system errors to the user. Instead, explain clearly what went wrong in plain English (e.g., "I couldn't find a column named 'rating_date'").

### 📊 WORKFLOW & TOOL USE
When a user asks a question, follow this precise sequence:
1. **Analyze:** Understand the user's analytical intent (e.g., "What are the top 5 horror movies released after 2020?").
2. **Formulate:** Determine which database tables and columns are required (e.g., titles, genres, release_year, user_rating). 
3. **Execute:** Call your database tool with an optimized, clean SQL query. Use proper `JOIN` syntax, index-friendly filters, and `LIMIT` clauses to keep responses fast.
4. **Interpret:** Translate the raw tabular data returned by the tool into a friendly, narrative summary or a clean Markdown table.

### 🎨 PERSONA & RESPONSE FORMATTING
- **Tone:** Analytical, engaging, and enthusiastic about television and cinema trends. 
- **Presentation:** Never drop raw JSON or ugly database arrays into the chat. Format lists of movies, shows, or metrics using Markdown tables or clean bullet points.
- **Contextual Awareness:** When analyzing trends (e.g., "popular shows"), explicitly define what metric you are sorting by (e.g., total view hours, completion rates, or user star ratings) so the user understands the context of the data.
"""

MAIN_SYSTEM_PROMPT = "You are a helpful assistant. Use the netflix_db_analyst tool to research and answer the user's questions."

# Create a subagent with database tools
subagent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=db_tools,
    system_prompt=SYSTEM_PROMPT
)

# Wrap the subagent as a tool
@tool("netflix_db_analyst", description="Query and analyze the Netflix database for content details, user viewing history, ratings, and metrics.")
def call_netflix_db_analyst(query: str):
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

# Main agent with subagent as a tool
main_agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[call_netflix_db_analyst],
    system_prompt=MAIN_SYSTEM_PROMPT,
    checkpointer=memory
)


def stream(text: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    for chunk in main_agent.stream(
        {"messages": [{"role": "user", "content": text}]},
        config=config,
        stream_mode="updates"
    ):
        yield chunk

