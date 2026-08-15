import pandas as pd
import re
import os

from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


app = Flask(__name__)
CORS(app)


STUDENT_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "student_performance.csv"
)

print("Loading student dataset...")

student_df = pd.read_csv(STUDENT_DATA_PATH)

print("Student dataset loaded.")
print("Total students:", len(student_df))
print("Student columns:", list(student_df.columns))


COLUMN_MAP = {
    "study hours": "StudyHours",
    "study hour": "StudyHours",
    "attendance": "Attendance",
    "resources": "Resources",
    "extracurricular": "Extracurricular",
    "motivation": "Motivation",
    "internet": "Internet",
    "gender": "Gender",
    "age": "Age",
    "learning style": "LearningStyle",
    "online courses": "OnlineCourses",
    "online course": "OnlineCourses",
    "discussions": "Discussions",
    "assignment completion": "AssignmentCompletion",
    "assignment completion rate": "AssignmentCompletion",
    "exam score": "ExamScore",
    "exam scores": "ExamScore",
    "edutech": "EduTech",
    "stress level": "StressLevel",
    "stress": "StressLevel",
    "final grade": "FinalGrade",
    "final grades": "FinalGrade"
}


def extract_age(question):
    question = question.lower()

    patterns = [
        r'age\s*(?:is|=|:)?\s*(\d+)',
        r'aged\s*(\d+)',
        r'(\d+)\s*year\s*old',
        r'(\d+)\s*years?\s*old'
    ]

    for pattern in patterns:
        match = re.search(pattern, question)

        if match:
            return int(match.group(1))

    return None


def extract_gender(question):
    question = question.lower()

    if re.search(r'\bfemale\b|\bgirl\b|\bwomen\b|\bwoman\b', question):
        return 0

    if re.search(r'\bmale\b|\bboy\b|\bmen\b|\bman\b', question):
        return 1

    return None


def extract_numeric_conditions(question):
    question = question.lower()
    conditions = []

    numeric_columns = {
        "study hours": "StudyHours",
        "study hour": "StudyHours",
        "attendance": "Attendance",
        "resources": "Resources",
        "extracurricular": "Extracurricular",
        "motivation": "Motivation",
        "internet": "Internet",
        "online courses": "OnlineCourses",
        "online course": "OnlineCourses",
        "discussions": "Discussions",
        "assignment completion": "AssignmentCompletion",
        "assignment completion rate": "AssignmentCompletion",
        "exam score": "ExamScore",
        "exam scores": "ExamScore",
        "examscore": "ExamScore",
        "edutech": "EduTech",
        "stress level": "StressLevel",
        "stress": "StressLevel",
        "final grade": "FinalGrade",
        "final grades": "FinalGrade"
    }

    operators = {
        "greater than or equal to": ">=",
        "more than or equal to": ">=",
        "less than or equal to": "<=",
        "at least": ">=",
        "at most": "<=",
        "greater than": ">",
        "more than": ">",
        "above": ">",
        "over": ">",
        "less than": "<",
        "below": "<",
        "under": "<",
        "equal to": "=",
        "equals": "=",
        "equal": "="
    }

    reverse_operator = {
        ">": "<",
        "<": ">",
        ">=": "<=",
        "<=": ">=",
        "=": "="
    }

    for name, column in sorted(
        numeric_columns.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):

        for phrase, operator in sorted(
            operators.items(),
            key=lambda x: len(x[0]),
            reverse=True
        ):

            patterns = [

                rf'\b{re.escape(name)}\s+'
                rf'{re.escape(phrase)}\s+'
                rf'(\d+(?:\.\d+)?)\b',

                rf'\b{re.escape(phrase)}\s+'
                rf'(\d+(?:\.\d+)?)\s+'
                rf'{re.escape(name)}\b',

                rf'\b(\d+(?:\.\d+)?)\s+'
                rf'{re.escape(phrase)}\s+'
                rf'{re.escape(name)}\b'
            ]

            for pattern_index, pattern in enumerate(patterns):

                matches = re.finditer(
                    pattern,
                    question
                )

                for match in matches:

                    value = float(match.group(1))

                    if pattern_index == 2:
                        final_operator = reverse_operator[operator]
                    else:
                        final_operator = operator

                    condition = (
                        column,
                        final_operator,
                        value
                    )

                    if condition not in conditions:
                        conditions.append(condition)

    symbol_pattern = re.compile(
        r'(?:(?P<column1>[a-zA-Z ]+?)\s*'
        r'(?P<operator1>>=|<=|>|<|=)\s*'
        r'(?P<value1>\d+(?:\.\d+)?)'
        r'|'
        r'(?P<value2>\d+(?:\.\d+)?)\s*'
        r'(?P<operator2>>=|<=|>|<|=)\s*'
        r'(?P<column2>[a-zA-Z ]+?))'
        r'(?=\s*(?:and|or|,|$))'
    )

    for match in symbol_pattern.finditer(question):

        if match.group("column1"):

            column_text = match.group("column1").strip()

            for name, column in sorted(
                numeric_columns.items(),
                key=lambda x: len(x[0]),
                reverse=True
            ):

                if name in column_text:

                    condition = (
                        column,
                        match.group("operator1"),
                        float(match.group("value1"))
                    )

                    if condition not in conditions:
                        conditions.append(condition)

                    break

        elif match.group("column2"):

            column_text = match.group("column2").strip()

            for name, column in sorted(
                numeric_columns.items(),
                key=lambda x: len(x[0]),
                reverse=True
            ):

                if name in column_text:

                    condition = (
                        column,
                        reverse_operator[
                            match.group("operator2")
                        ],
                        float(match.group("value2"))
                    )

                    if condition not in conditions:
                        conditions.append(condition)

                    break

    return conditions


def extract_level_conditions(question):
    question = question.lower()
    conditions = []

    level_map = {
        "low": 0,
        "medium": 1,
        "moderate": 1,
        "high": 2
    }

    level_columns = {
        "motivation": "Motivation",
        "stress level": "StressLevel",
        "stress": "StressLevel",
        "resources": "Resources",
        "extracurricular": "Extracurricular",
        "internet": "Internet",
        "edutech": "EduTech"
    }

    for name, column in sorted(
        level_columns.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):

        for level, value in level_map.items():

            patterns = [
                rf'\b{level}\s+{re.escape(name)}\b',
                rf'\b{re.escape(name)}\s+'
                rf'(?:is|=|of)?\s*{level}\b'
            ]

            if any(
                re.search(pattern, question)
                for pattern in patterns
            ):
                condition = (
                    column,
                    "=",
                    value
                )

                if condition not in conditions:
                    conditions.append(condition)

    return conditions


def extract_conditions(question):
    conditions = []

    conditions.extend(
        extract_numeric_conditions(question)
    )

    conditions.extend(
        extract_level_conditions(question)
    )

    return conditions


def filter_students(question):

    result = student_df.copy()

    age = extract_age(question)
    gender = extract_gender(question)

    if age is not None:
        result = result[
            result["Age"] == age
        ]

    if gender is not None:
        result = result[
            result["Gender"] == gender
        ]

    conditions = extract_conditions(question)

    for column, operator, value in conditions:

        if operator == ">":
            result = result[
                result[column] > value
            ]

        elif operator == ">=":
            result = result[
                result[column] >= value
            ]

        elif operator == "<":
            result = result[
                result[column] < value
            ]

        elif operator == "<=":
            result = result[
                result[column] <= value
            ]

        elif operator == "=":
            result = result[
                result[column] == value
            ]

    return result


def extract_requested_columns(question):
    question = question.lower()
    requested = []

    phrases = sorted(
        COLUMN_MAP.keys(),
        key=len,
        reverse=True
    )

    for phrase in phrases:

        if phrase not in question:
            continue

        column = COLUMN_MAP[phrase]

        if column in ["Age", "Gender"]:
            continue

        filter_patterns = [

            rf'\b{re.escape(phrase)}\s+'
            rf'(above|over|more than|greater than|below|under|'
            rf'less than|at least|at most|equal to|equals|equal)'
            rf'\s+\d+(?:\.\d+)?',

            rf'\b(above|over|more than|greater than|below|under|'
            rf'less than|at least|at most|equal to|equals|equal)'
            rf'\s+\d+(?:\.\d+)?\s+'
            rf'{re.escape(phrase)}',

            rf'\b{re.escape(phrase)}\s*'
            rf'(>=|<=|>|<|=)\s*'
            rf'\d+(?:\.\d+)?',

            rf'\b\d+(?:\.\d+)?\s*'
            rf'(>=|<=|>|<|=)\s*'
            rf'{re.escape(phrase)}'
        ]

        if any(
            re.search(
                pattern,
                question
            )
            for pattern in filter_patterns
        ):
            continue

        if column == "Motivation":

            if re.search(
                r'\b(low|medium|moderate|high)\s+motivation\b',
                question
            ):
                continue

        if column == "StressLevel":

            if re.search(
                r'\b(low|medium|moderate|high)\s+stress'
                r'(?:\s+level)?\b',
                question
            ):
                continue

        if column not in requested:
            requested.append(column)

    return requested


def is_average_question(question):

    return any(
        word in question.lower()
        for word in [
            "average",
            "mean",
            "avg"
        ]
    )


def is_highest_question(question):

    return any(
        word in question.lower()
        for word in [
            "highest",
            "maximum",
            "max"
        ]
    )


def is_lowest_question(question):

    return any(
        word in question.lower()
        for word in [
            "lowest",
            "minimum",
            "min"
        ]
    )


def is_count_question(question):

    question = question.lower()

    return (
        "how many" in question
        or "number of" in question
        or "count of" in question
    )


def calculate_student_answer(question, data):

    if data.empty:
        return "No students matched the given conditions."

    requested_columns = extract_requested_columns(question)

    if is_count_question(question):

        return (
            f"There are {len(data)} students "
            f"matching the given conditions."
        )

    if is_average_question(question):

        if not requested_columns:
            return (
                "Please specify what you want the average of."
            )

        target_column = requested_columns[0]

        value = data[target_column].mean()

        display_names = {
            "StudyHours": "study hours",
            "Attendance": "attendance",
            "Resources": "resources",
            "Extracurricular": "extracurricular",
            "Motivation": "motivation",
            "Internet": "internet",
            "OnlineCourses": "online courses",
            "Discussions": "discussions",
            "AssignmentCompletion": "assignment completion",
            "ExamScore": "exam score",
            "EduTech": "edutech",
            "StressLevel": "stress level",
            "FinalGrade": "final grade"
        }

        display_name = display_names.get(
            target_column,
            target_column
        )

        return (
            f"The average {display_name} "
            f"of the matching students is {value:.2f}."
        )

    if is_highest_question(question):

        if not requested_columns:
            return (
                "Please specify what you want "
                "the highest value of."
            )

        target_column = requested_columns[0]

        value = data[target_column].max()

        return (
            f"The highest {target_column} "
            f"of the matching students is {value}."
        )

    if is_lowest_question(question):

        if not requested_columns:
            return (
                "Please specify what you want "
                "the lowest value of."
            )

        target_column = requested_columns[0]

        value = data[target_column].min()

        return (
            f"The lowest {target_column} "
            f"of the matching students is {value}."
        )

    if requested_columns:

        return data[
            requested_columns
        ].to_string(index=False)

    return data.to_string(index=False)


def is_student_question(question):

    question = question.lower()

    student_keywords = [
        "student",
        "students",
        "study hours",
        "study hour",
        "attendance",
        "motivation",
        "exam score",
        "exam scores",
        "final grade",
        "final grades",
        "assignment completion",
        "online courses",
        "online course",
        "learning style",
        "stress level",
        "stress",
        "extracurricular",
        "discussions",
        "edutech",
        "age",
        "gender",
        "internet",
        "resources"
    ]

    return any(
        keyword in question
        for keyword in student_keywords
    )


print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


print("Loading ChromaDB...")

vector_db = Chroma(
    persist_directory=os.path.join(
    os.path.dirname(__file__),
    "chroma_db"
    ),
    embedding_function=embedding_model
)

print("ChromaDB loaded successfully.")


print("Loading Ollama LLM...")

llm = OllamaLLM(
    model="llama3.1",
    temperature=0
)

print("Ollama LLM loaded.")


def get_answer(question):

    if is_student_question(question):

        data = filter_students(question)

        answer = calculate_student_answer(
            question,
            data
        )

        return answer, []


    docs = vector_db.similarity_search(
        question,
        k=5
    )

    context_parts = []

    for i, doc in enumerate(docs, start=1):

        source = doc.metadata.get(
            "source",
            "Unknown source"
        )

        context_parts.append(
            f"""
DOCUMENT {i}
SOURCE: {source}

{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an AI medical information assistant for a system
that contains knowledge about:

1. Lung Cancer
2. Mango Diseases

Answer the user's question using ONLY the information
available in the CONTEXT below.

Rules:

1. Use the retrieved documents as the primary source.
2. If the question is about mango diseases, use mango
   information from the context.
3. If the question is about lung cancer, use lung cancer
   information from the context.
4. Do not mix unrelated topics.
5. Do not invent facts.
6. If the context does not contain enough information,
   clearly say that the information was not found in the
   available knowledge base.
7. Do not diagnose the user.
8. Do not predict whether a person has cancer.
9. Give a clear and focused answer.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response, docs


@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        if not data or "question" not in data:

            return jsonify({
                "answer": "Please enter a question.",
                "sources": []
            }), 400

        question = data["question"].strip()

        if not question:

            return jsonify({
                "answer": "Please enter a question.",
                "sources": []
            }), 400

        print("\n===================================")
        print("QUESTION:", question)
        print("===================================")

        answer, docs = get_answer(question)

        sources = []

        if docs:

            for doc in docs:

                source = doc.metadata.get(
                    "source",
                    "Unknown source"
                )

                if source not in sources:
                    sources.append(source)

        elif is_student_question(question):

            sources = [
                "Student Performance Dataset"
            ]

        print("\nANSWER:")
        print(answer)

        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:

        print("\nERROR:", e)

        return jsonify({
            "answer": "Unable to generate answer.",
            "sources": [],
            "error": str(e)
        }), 500


if __name__ == "__main__":

    print("\n===================================")
    print("RAG API SERVER STARTING")
    print("===================================")
    print("Knowledge Base: Lung Cancer + Mango Diseases")
    print("Student Dataset: Student Performance")
    print(
        "ChromaDB:",
        os.path.join(
        os.path.dirname(__file__),
        "chroma_db"
       )
    )
    print("API: http://127.0.0.1:5000")
    print("===================================\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
