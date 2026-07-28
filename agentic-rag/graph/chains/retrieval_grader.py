from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(temperature=0, model = "gemini-2.5-flash")

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes', or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """
    You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keywords or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' to indicate wheter the document is relevant to the question that is asked.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User Question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

