from flask import Flask, request, jsonify
from flask_cors import CORS
from student_rag import query_students

app = Flask(__name__)
CORS(app)

@app.route("/student-rag", methods=["POST"])
def student_rag():

    try:
        data = request.get_json()

        question = data.get("question", "").strip()

        if not question:
            return jsonify({
                "error": "Question is required"
            }), 400

        result = query_students(question)

        return jsonify({
            "answer": result["answer"],
            "count": result["count"]
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    print("Student RAG API running on port 5002...")
    app.run(host="127.0.0.1", port=5002, debug=False)