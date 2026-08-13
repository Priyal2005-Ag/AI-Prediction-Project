from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load embedding model

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load existing Chroma database

vector_db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


# User question

question = input("Ask your question: ")


# Search relevant chunks

results = vector_db.similarity_search_with_score(
    question,
    k=5
)


print("\n==============================")
print("RELEVANT INFORMATION")
print("==============================\n")


for i, (doc, score) in enumerate(results):

    print("------------------------------")
    print("Result:", i+1)
    print("Source:", doc.metadata.get("source"))
    print("Score:", score)
    print()

    print(doc.page_content[:500])
    print()