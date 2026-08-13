from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_documents import documents


# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)


# Split documents
chunks = text_splitter.split_documents(documents)


print("==============================")
print("CHUNKING COMPLETE")
print("==============================")

print("Original documents:", len(documents))
print("Total chunks:", len(chunks))


# Preview chunks

for i, chunk in enumerate(chunks[:5]):

    print("\n-----------------------")
    print("Chunk", i+1)
    print("Source:", chunk.metadata.get("source"))
    print("Characters:", len(chunk.page_content))
    print(chunk.page_content[:300])