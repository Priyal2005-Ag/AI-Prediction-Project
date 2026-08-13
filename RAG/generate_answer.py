from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


# -----------------------------
# Load Embedding Model
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# Load ChromaDB
# -----------------------------

vector_db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


# -----------------------------
# Load Local LLM
# -----------------------------

llm = OllamaLLM(
    model="llama3.1"
)


# -----------------------------
# Ask Question
# -----------------------------

question = input("Ask your question: ")


# -----------------------------
# Retrieve Documents
# -----------------------------

results = vector_db.similarity_search(
    question,
    k=5
)


# -----------------------------
# Combine Context
# -----------------------------

context = "\n\n".join(
    [
        doc.page_content
        for doc in results
    ]
)


# -----------------------------
# Prompt
# -----------------------------

prompt = f"""
You are an AI assistant for a disease information system.

The knowledge base contains information about:
1. Lung cancer
2. Mango diseases

Answer the user's question using ONLY the information provided in the context.

Do not use outside knowledge.
Do not make up information.
If the answer cannot be found in the provided context, say:
"The available knowledge base does not contain enough information to answer this question."

Identify the topic from the retrieved context and answer accordingly.

Context:
{context}

Question:
{question}

Answer:
"""


# -----------------------------
# Generate Answer
# -----------------------------

response = llm.invoke(prompt)


print("\n==============================")
print("AI ANSWER")
print("==============================\n")

print(response)