from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.embeddings.embedding_model import model


def create_vector_store(chunks):
    documents = []

    for i, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "source": "Uploaded PDF",
                    "page": chunk["page"],
                    "chunk_id": i + 1,
                    "length": len(chunk["text"])
                }
            )
        )

    vector_store = FAISS.from_documents(
        documents,
        model
    )

    return vector_store


def retrieve_chunks(vector_store, query):

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.6
        }
    )

    docs = retriever.invoke(query)

    return docs