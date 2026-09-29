import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    GEMINI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from src.embeddings import GeminiEmbeddings


# Load environment variables
load_dotenv()


# ---------------------------------------------------------
# PDF PATH
# ---------------------------------------------------------

PDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "Ebook-Agentic-AI.pdf"
)


# ---------------------------------------------------------
# LOAD AND SPLIT PDF
# ---------------------------------------------------------

def load_and_split_pdf():

    print("Loading PDF...")

    loader = PyPDFLoader(PDF_PATH)

    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    # Add useful metadata
    for i, chunk in enumerate(chunks):

        chunk.metadata["source"] = "Ebook-Agentic-AI.pdf"

        chunk.metadata["chunk_id"] = i

        # PyPDFLoader page numbers start from 0
        chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1

    print(f"Created {len(chunks)} chunks.")

    return chunks


# ---------------------------------------------------------
# CREATE / CONNECT TO PINECONE
# ---------------------------------------------------------

def create_pinecone_index():

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [
        index["name"]
        for index in pc.list_indexes()
    ]

    if PINECONE_INDEX_NAME not in existing_indexes:

        print(
            f"Creating Pinecone index: "
            f"{PINECONE_INDEX_NAME}"
        )

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print("Pinecone index created.")

        # Give Pinecone some time to initialize
        time.sleep(10)

    else:

        print(
            f"Pinecone index already exists: "
            f"{PINECONE_INDEX_NAME}"
        )

    return pc


# ---------------------------------------------------------
# INGEST DOCUMENTS
# ---------------------------------------------------------

def ingest_documents():

    # Load and split PDF
    chunks = load_and_split_pdf()

    # Connect to Pinecone
    pc = create_pinecone_index()

    index = pc.Index(PINECONE_INDEX_NAME)

    # Gemini embedding model
    embeddings = GeminiEmbeddings()

    print()
    print("Creating Gemini embeddings...")
    print()

    # -----------------------------------------------------
    # IMPORTANT
    # First 100 chunks were already uploaded successfully.
    # We continue from chunk 100.
    # -----------------------------------------------------

    START_FROM = 100

    # Smaller batch helps reduce API quota pressure
    batch_size = 10

    total_uploaded = START_FROM

    print(
        f"Resuming ingestion from chunk "
        f"{START_FROM}..."
    )

    print(
        f"Remaining chunks: "
        f"{len(chunks) - START_FROM}"
    )

    print()

    # Start from chunk 100 instead of 0
    for start in range(
        START_FROM,
        len(chunks),
        batch_size
    ):

        batch = chunks[
            start:start + batch_size
        ]

        texts = [
            chunk.page_content
            for chunk in batch
        ]

        # -------------------------------------------------
        # Create Gemini embeddings
        # -------------------------------------------------

        print(
            f"Creating embeddings for "
            f"chunks {start} - "
            f"{start + len(batch) - 1}..."
        )

        try:

            vectors = embeddings.embed_documents(
                texts
            )

        except Exception as e:

            print()
            print(
                "ERROR while creating embeddings:"
            )

            print(e)

            print()
            print(
                "If this is a Gemini 429 quota error, "
                "wait 30-60 seconds and run the command again."
            )

            print(
                "The script will resume from chunk 100."
            )

            raise

        # -------------------------------------------------
        # Prepare Pinecone records
        # -------------------------------------------------

        records = []

        for chunk, vector in zip(
            batch,
            vectors
        ):

            records.append(
                {
                    "id": (
                        f"chunk-"
                        f"{chunk.metadata['chunk_id']}"
                    ),

                    "values": vector,

                    "metadata": {
                        "text": chunk.page_content,

                        "source": (
                            chunk.metadata["source"]
                        ),

                        "page": (
                            chunk.metadata["page"]
                        ),

                        "chunk_id": (
                            chunk.metadata["chunk_id"]
                        )
                    }
                }
            )

        # -------------------------------------------------
        # Upload to Pinecone
        # -------------------------------------------------

        index.upsert(
            vectors=records
        )

        total_uploaded += len(records)

        print(
            f"Uploaded "
            f"{total_uploaded}/{len(chunks)} chunks..."
        )

        print()

        # Small delay to reduce rate-limit pressure
        time.sleep(2)

    # -----------------------------------------------------
    # SUCCESS MESSAGE
    # -----------------------------------------------------

    print()
    print("=" * 50)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 50)

    print(
        f"Pages processed: "
        f"{len(set(c.metadata['page'] for c in chunks))}"
    )

    print(
        f"Total chunks: "
        f"{len(chunks)}"
    )

    print(
        f"Chunks uploaded: "
        f"{total_uploaded}"
    )

    print(
        f"Pinecone index: "
        f"{PINECONE_INDEX_NAME}"
    )

    print("=" * 50)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    ingest_documents()