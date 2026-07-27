import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load env variables
load_dotenv()

# Initialize Google GenAI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    output_dimensionality=1024
)

# Initialize the vectorstore
vectorstore = PineconeVectorStore(
    index_name="docs", 
    embedding=embeddings,
    pinecone_api_key=os.environ.get("PINECONE_API_KEY")
)

# Initialize the chat model
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain"""

    # Retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 8}).invoke(query)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized content and retrieved docs artifact
    return serialized, retrieved_docs

def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    """

    system_prompt = (
        "You are helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Provide detailed, comprehensive explanations with code examples or bullet points where relevant. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    messages = [{"role": "user", "content": query}]

    response = agent.invoke({"messages": messages})

    # Extract clean text from the last AI message
    last_message = response["messages"][-1]
    
    if hasattr(last_message, "text") and last_message.text:
        answer = last_message.text
    else:
        raw_content = last_message.content
        if isinstance(raw_content, str):
            answer = raw_content
        elif isinstance(raw_content, list):
            text_parts = []
            for block in raw_content:
                if isinstance(block, dict):
                    text_parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    text_parts.append(getattr(block, "text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            answer = "\n".join(filter(None, text_parts))
        else:
            answer = str(raw_content)

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == "__main__":
    result = run_llm(query="what are deep agents?")
    print(result)