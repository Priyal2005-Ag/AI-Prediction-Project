import requests

API_URL = "http://127.0.0.1:5000/ask"

print("\n===================================")
print("        RAG CHAT SYSTEM")
print("===================================")
print("Knowledge Base: Lung Cancer + Mango Diseases")
print("Type 'exit' to quit.")
print("===================================\n")

while True:

    question = input("Ask your question: ")

    if question.lower() == "exit":
        print("\nRAG Chat closed.")
        break

    if not question.strip():
        print("Please enter a question.\n")
        continue

    try:

        response = requests.post(
            API_URL,
            json={
                "question": question
            }
        )

        if response.status_code == 200:

            data = response.json()

            print("\n==============================")
            print("AI ANSWER")
            print("==============================")

            print(data.get("answer", "No answer received."))

            print("\nSOURCES:")
            for source in data.get("sources", []):
                print("-", source)

            print()

        else:

            print("\nAPI ERROR")
            print(response.text)
            print()

    except requests.exceptions.ConnectionError:

        print("\nERROR: Cannot connect to RAG API.")
        print("Make sure app_api.py is running.")
        print()

    except Exception as e:

        print("\nERROR:", e)
        print()