# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly from the provided **Agentic AI for Executives** eBook.

The system uses LangGraph to orchestrate the RAG workflow, Pinecone for vector search, Gemini embeddings for semantic retrieval, Groq GPT-OSS for answer generation, and FastAPI to expose the chatbot through an API.

---

## Project Objective

The objective of this project is to build a grounded RAG chatbot that:

- Retrieves relevant information from the Agentic AI eBook.
- Generates answers only from the retrieved context.
- Avoids using outside knowledge.
- Refuses questions when the required information is not available in the eBook.
- Returns retrieved context chunks and similarity scores.
- Provides a confidence score for each response.
- Exposes the RAG pipeline through a FastAPI endpoint.

---

## Architecture

```text
Agentic AI eBook PDF
        |
        v
   PDF Loading
        |
        v
 Text Chunking
        |
        v
 Gemini Embeddings
        |
        v
    Pinecone
 Vector Database
        |
        v
    User Query
        |
        v
 Gemini Query Embedding
        |
        v
 Top-K Similarity Search
        |
        v
     LangGraph
        |
        v
 Retrieved Context
        |
        v
 Groq GPT-OSS 20B
        |
        v
 Grounded Answer
        |
        v
 Confidence Score
        |
        v
     FastAPI