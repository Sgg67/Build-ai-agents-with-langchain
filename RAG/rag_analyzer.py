# import necessary modules
import os
from dotenv import load_dotenv
from google import genai
from operator import itemgetter
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

 # intialize google gen ai embeddings object
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    output_dimensionality=1024
     )

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

vectorstore = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX"], embedding=embeddings
)

 # retrieve the content from the vector db
retriever = vectorstore.as_retriever(search_kwargs={"k" : 3})

prompt_template = ChatPromptTemplate.from_template(
"""
    Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:

"""
)

# implementation 1: mwithout lcel
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL
    Manually retrieves documents, formats them, and generates a response
    """

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    # format documents into a context string
    context = format_docs(docs)

    # format the prompt with context and and question
    messages = prompt_template.format_messages(context=context, question=query)

    # invoke llm with formatting messages
    response = llm.invoke(messages)

    # return the content of the response
    return response.content

# implement second method that uses lcel
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL(LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question")  | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("retrieving...")

    query= "what is Pinecone in machine learning?"

    # option 0 raw invocation without RAG
    result_raw = llm.invoke([HumanMessage(content=query)])
    print(result_raw.content)

    # option 1 use implementation without LCEL
    result_without_lcel = retrieval_chain_without_lcel(query)
    print(result_without_lcel)

    # option 2 use lcel ( langchain expression language)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question" : query})
    print(result_with_lcel)
   