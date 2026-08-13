from chunk_documents import chunks
from sentence_transformers import SentenceTransformer
import pickle


# Load embedding model
print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Extract text from chunks
texts = [
    chunk.page_content
    for chunk in chunks
]


print("Generating embeddings...")


# Create embeddings
embeddings = model.encode(
    texts,
    show_progress_bar=True
)


# Save embeddings and chunks

with open("embeddings.pkl", "wb") as f:
    pickle.dump(
        embeddings,
        f
    )


with open("chunks.pkl", "wb") as f:
    pickle.dump(
        chunks,
        f
    )


print("==============================")
print("EMBEDDING COMPLETE")
print("==============================")

print(
    "Total embeddings:",
    len(embeddings)
)

print(
    "Embedding size:",
    len(embeddings[0])
)