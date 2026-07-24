# import necessary modules
from typing import List
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from tavily import TavilyClient
from langchain_tavily import TavilySearch

# add a source class for te source of the information
class Source(BaseModel):
    """Schema for a source used by an agent"""

    url: str = Field(description="The URL of the source")

# add a response class with the agent response
class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the query ")
    temp_f: float = Field(description= "The temperature in farnheit extracted from the query" )
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")


tavily = TavilyClient()
# load the env file
load_dotenv()

llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash-lite")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    result = agent.invoke({"messages": HumanMessage(content="What is the weather in Tokyo?")})
    response: AgentResponse = result["structured_response"]
    print(f"Temperature is: {response.temp_f} farnheit")
    
if __name__ == "__main__":
    main()

