import asyncio
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

load_dotenv()

# Build an absolute path safely based on the current file directory
SERVER_PATH = Path(__file__).parent / "servers" / "math_server.py"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

studio_server_params = StdioServerParameters(
    command="python",
    args=[str(SERVER_PATH)],
)

async def main():
    async with stdio_client(studio_server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("session initialized")
            tools = await load_mcp_tools(session)
            print(tools)
            agent = create_agent(llm, tools)
            result = await agent.ainvoke({"messages": [HumanMessage(content="what is 2 + 2?")]})
            print(result["messages"][-1].content)
if __name__ == "__main__":
    asyncio.run(main())