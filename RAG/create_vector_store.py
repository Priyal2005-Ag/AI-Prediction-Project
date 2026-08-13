from chunk_documents import chunks
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


print("Loading embedding model...")


# Same embedding model used earlier
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


print("Creating ChromaDB...")


# Create and store embeddings
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)


print("==============================")
print("CHROMADB CREATED SUCCESSFULLY")
print("==============================")

print("Total chunks stored:", len(chunks))