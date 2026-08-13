import os
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


# ============================================================
# KNOWLEDGE BASE PATH
# ============================================================

KNOWLEDGE_BASE = r"C:\Users\Lenovo\OneDrive\RAG\Knowledge base"


# ============================================================
# DOCUMENT LIST
# ============================================================

documents = []


# ============================================================
# CHECK FOLDER
# ============================================================

if not os.path.exists(KNOWLEDGE_BASE):

    print("Knowledge base folder not found!")
    exit()


# ============================================================
# LOAD ALL FILES INCLUDING SUBFOLDERS
# ============================================================

for root, dirs, files in os.walk(KNOWLEDGE_BASE):

    for filename in files:

        filepath = os.path.join(root, filename)

        # Relative folder path
        relative_folder = os.path.relpath(
            root,
            KNOWLEDGE_BASE
        )


        # ====================================================
        # DETERMINE TOPIC
        # ====================================================

        if relative_folder == ".":

            # Existing files directly inside Knowledge base
            topic = "lung_cancer"

        elif relative_folder.lower() == "mango_disease":

            topic = "mango_disease"

        else:

            topic = relative_folder.lower().replace("\\", "/")


        # ====================================================
        # PDF FILES
        # ====================================================

        if filename.lower().endswith(".pdf"):

            print(
                f"Loading PDF: {filepath}"
            )

            try:

                loader = PyPDFLoader(
                    filepath
                )

                pdf_documents = loader.load()


                # Add metadata
                for doc in pdf_documents:

                    doc.metadata["source"] = filename

                    doc.metadata["topic"] = topic

                    doc.metadata["file_path"] = filepath


                documents.extend(
                    pdf_documents
                )


            except Exception as e:

                print(
                    f"Skipping PDF: {filename}"
                )

                print(
                    "Reason:",
                    e
                )


        # ====================================================
        # HTML FILES
        # ====================================================

        elif filename.lower().endswith(".html"):

            print(
                f"Loading HTML: {filepath}"
            )

            try:

                with open(
                    filepath,
                    "r",
                    encoding="utf-8"
                ) as file:

                    html_content = file.read()


                soup = BeautifulSoup(
                    html_content,
                    "lxml"
                )


                # Remove unwanted elements
                for tag in soup(
                    [
                        "script",
                        "style",
                        "nav",
                        "footer"
                    ]
                ):

                    tag.decompose()


                text = soup.get_text(
                    separator=" ",
                    strip=True
                )


                html_document = Document(

                    page_content=text,

                    metadata={

                        "source": filename,

                        "topic": topic,

                        "file_path": filepath

                    }

                )


                documents.append(
                    html_document
                )


            except Exception as e:

                print(
                    f"Skipping HTML: {filename}"
                )

                print(
                    "Reason:",
                    e
                )


# ============================================================
# FINAL RESULT
# ============================================================

print(
    "\n=============================="
)

print(
    "DOCUMENT LOADING COMPLETE"
)

print(
    "=============================="
)


print(
    f"Total documents loaded: {len(documents)}"
)


# ============================================================
# SHOW TOPIC COUNTS
# ============================================================

topic_counts = {}


for doc in documents:

    topic = doc.metadata.get(
        "topic",
        "unknown"
    )

    topic_counts[topic] = (
        topic_counts.get(topic, 0) + 1
    )


print(
    "\nDocuments by topic:"
)

for topic, count in topic_counts.items():

    print(
        f"{topic}: {count}"
    )


# ============================================================
# PREVIEW
# ============================================================

for i, doc in enumerate(
    documents[:5]
):

    print(
        "\n------------------------------"
    )

    print(
        f"Document {i + 1}"
    )

    print(
        "Source:",
        doc.metadata.get("source")
    )

    print(
        "Topic:",
        doc.metadata.get("topic")
    )

    print(
        "Characters:",
        len(doc.page_content)
    )

    print(
        doc.page_content[:300]
    )