from dotenv import load_dotenv
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from react import llm, tools
from langchain_core.messages import SystemMessage

load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assistant. 
When asked to perform multi-step tasks (e.g., finding information and then calculating a number):
1. Use search tools first to find the factual data/number.
2. ALWAYS use the mathematical calculation tools provided to compute final results rather than doing mental math.
"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    messages = [SystemMessage(content=SYSTEM_MESSAGE)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)


