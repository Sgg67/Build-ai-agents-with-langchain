from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Initialize client and explicitly allow public prompt pull
client = Client()
prompt = client.pull_prompt(
    "rlm/rag-prompt", 
    dangerously_pull_public_prompt=True
)

generation_chain = prompt | llm | StrOutputParser()