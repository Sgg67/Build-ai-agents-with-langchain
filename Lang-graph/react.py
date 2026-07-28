from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

# create a tool that triples a number
@tool
def triple(num: float) -> float:
    """
    Triples the given input number. Pass the numeric value of the temperature here.
    """
    return float(num) * 3


# set the tools 
tools = [TavilySearch(max_results=1), triple]

# set up the llm with the tools
llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash").bind_tools(tools)


