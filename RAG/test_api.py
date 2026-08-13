import requests


response = requests.post(
    "http://127.0.0.1:5000/ask",
    json={
        "question":
        "What are the symptoms of mango powdery mildew?"
    }
)


print(response.json())