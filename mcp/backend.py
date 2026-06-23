# https://www.alltrails.com/mcp

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

async def main():
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",  # HTTP-based remote server
                "url": "https://www.alltrails.com/mcp",
            }
        }
    )

    tools = await client.get_tools()
    # for tool in tools:
    #     print(tool.name)
    #     print(tool.description)
    #     # print(tool.args)
    #     print("---\n")
    agent = create_agent(
        llm,
        tools  
    )
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "which hikes can you suggest in israel?"}]}
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())