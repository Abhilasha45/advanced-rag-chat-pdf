from langchain_huggingface import HuggingFaceEmbeddings

model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.embed_documents(texts)
    return embeddings