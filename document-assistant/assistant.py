import asyncio
import os
from typing import List
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl

# Load env variables
load_dotenv()

# Initialize Google GenAI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    output_dimensionality=1024
)

vectorstore = PineconeVectorStore(
    index_name="docs", 
    embedding=embeddings,
    pinecone_api_key=os.environ.get("PINECONE_API_KEY")
)
tavily_crawl = TavilyCrawl()

def create_batches(documents: List[Document], batch_size: int = 50) -> List[List[Document]]:
    """Split a list of documents into batches."""
    return [documents[i: i + batch_size] for i in range(0, len(documents), batch_size)]

async def upload_batch(batch: List[Document], batch_num: int) -> bool:
    """Upload a single batch asynchronously to Pinecone."""
    try:
        await vectorstore.aadd_documents(batch)
        print(f"Batch {batch_num} successfully uploaded.")
        return True
    except Exception as e:
        print(f"Failed to upload batch {batch_num}: {e}")
        return False

async def main():
    print("Crawling site...")
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 4,
        "extract_depth": "advanced",
    })

    all_docs = [
        Document(
            page_content=result.get('raw_content', ''), 
            metadata={"source": result.get('url', '')}
        )
        for result in res.get('results', [])
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    print(f"Total chunks created: {len(splitted_docs)}")

    batches = create_batches(splitted_docs, batch_size=50)

    # Sequentially process batches to prevent aiohttp session closures
    successful = 0
    for i, batch in enumerate(batches, start=1):
        if await upload_batch(batch, i):
            successful += 1

    print(f"Finished: {successful}/{len(batches)} batches successfully sent to Pinecone.")

if __name__ == "__main__":
    asyncio.run(main())