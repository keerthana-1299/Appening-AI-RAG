from typing import TypedDict

from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone

from src.config import (
    GROQ_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    TOP_K,
)

from src.embeddings import GeminiEmbeddings
from src.prompts import build_prompt

from langgraph.graph import StateGraph, START, END


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# AGENT STATE
# =========================================================

class AgentState(TypedDict):
    question: str
    context: list
    answer: str
    confidence_score: float


# =========================================================
# CLIENTS
# =========================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)

pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY
)

pinecone_index = pinecone_client.Index(
    PINECONE_INDEX_NAME
)

embedding_model = GeminiEmbeddings()


# =========================================================
# RETRIEVE INFORMATION FROM PINECONE
# =========================================================

def retrieve(state):

    question = state["question"]

    print(
        f"\nRetrieving information for: "
        f"{question}"
    )

    # Create embedding for user question
    query_vector = embedding_model.embed_query(
        question
    )

    # Search Pinecone
    results = pinecone_index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True
    )

    context = []

    for match in results["matches"]:

        metadata = match.get(
            "metadata",
            {}
        )

        context.append(
            {
                "text": metadata.get(
                    "text",
                    ""
                ),

                "page": metadata.get(
                    "page",
                    ""
                ),

                "source": metadata.get(
                    "source",
                    "Ebook-Agentic-AI.pdf"
                ),

                "score": float(
                    match.get(
                        "score",
                        0
                    )
                )
            }
        )

    print(
        f"Retrieved {len(context)} chunks."
    )

    return {
        "context": context
    }


# =========================================================
# GENERATE GROUNDED ANSWER
# =========================================================

def generate_answer(state):

    question = state["question"]

    context = state["context"]

    # -----------------------------------------------------
    # No relevant context
    # -----------------------------------------------------

    if not context:

        return {
            "answer": (
                "I could not find relevant information "
                "in the Agentic AI eBook."
            ),

            "confidence_score": 0.0
        }

    # -----------------------------------------------------
    # Prepare context
    # -----------------------------------------------------

    context_text = "\n\n".join(
        [
            (
                f"[Page {item['page']}]\n"
                f"{item['text']}"
            )
            for item in context
        ]
    )

    # -----------------------------------------------------
    # Build strict RAG prompt
    # -----------------------------------------------------

    prompt = build_prompt(
        question,
        context_text
    )

    # -----------------------------------------------------
    # Groq generation
    # -----------------------------------------------------

    print(
        "Generating answer using Groq..."
    )

    response = groq_client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict RAG assistant. "
                    "Answer ONLY using the provided "
                    "Agentic AI eBook context. "
                    "Do not use outside knowledge. "
                    "Do not invent information. "
                    "If the answer is not present in "
                    "the context, say that the information "
                    "is not available in the Agentic AI "
                    "eBook."
                )
            },

            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.0,

        max_tokens=500
    )

    answer = response.choices[0].message.content

    # -----------------------------------------------------
    # Calculate confidence from retrieval similarity
    # -----------------------------------------------------

    scores = [
        item["score"]
        for item in context
    ]

    confidence = round(
        max(
            0.0,
            min(
                1.0,
                sum(scores) / len(scores)
            )
        ),
        2
    )

    print(
        f"Confidence score: {confidence}"
    )

    return {
        "answer": answer,
        "confidence_score": confidence
    }


# =========================================================
# BUILD LANGGRAPH WORKFLOW
# =========================================================

workflow = StateGraph(
    AgentState
)


workflow.add_node(
    "retrieve",
    retrieve
)


workflow.add_node(
    "generate",
    generate_answer
)


workflow.add_edge(
    START,
    "retrieve"
)


workflow.add_edge(
    "retrieve",
    "generate"
)


workflow.add_edge(
    "generate",
    END
)


rag_graph = workflow.compile()


# =========================================================
# ASK QUESTION
# =========================================================

def ask_question(
    question: str
):

    return rag_graph.invoke(
        {
            "question": question,

            "context": [],

            "answer": "",

            "confidence_score": 0.0
        }
    )