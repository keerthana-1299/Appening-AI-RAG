from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import ask_question


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description=(
        "RAG chatbot grounded strictly "
        "in the Agentic AI eBook."
    ),
    version="1.0.0",
)


class ChatRequest(BaseModel):

    query: str


@app.get("/")
def root():

    return {
        "message": "Agentic AI RAG Chatbot is running."
    }


@app.post("/chat")
def chat(request: ChatRequest):

    result = ask_question(
        request.query
    )

    retrieved_chunks = []

    for item in result["context"]:

        retrieved_chunks.append(
            {
                "text": item["text"],
                "page": item["page"],
                "source": item["source"],
                "similarity_score": item["score"],
            }
        )

    return {
        "query": request.query,
        "final_answer": result["answer"],
        "retrieved_context_chunks": retrieved_chunks,
        "confidence_score": result[
            "confidence_score"
        ],
    }