# import necessary modules
import os
from dotenv import load_dotenv
from google import genai
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()

if __name__ == "__main__":

    print("Ingesting: ")
    loader = UnstructuredLoader(file_path="C:/Users/sagey/Downloads/Build-ai-agents-with-langchain/RAG/context.txt", chunking_strategy="basic", max_characters=1000000)
    document = loader.load()

    print("Splitting: ")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # intialize google gen ai embeddings object
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        output_dimensionality=1024
    )

    print("ingesting..")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["PINECONE_INDEX"])
    print("Finish")