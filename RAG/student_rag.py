import pandas as pd
import re

DATA_PATH = r"C:\Users\Lenovo\OneDrive\AI Prediction project\RAG\student_performance.csv"

df = pd.read_csv(DATA_PATH)

print("Student dataset loaded successfully.")
print("Total students:", len(df))
print("Columns:", list(df.columns))


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
        "attendance": "Attendance",
        "resources": "Resources",
        "extracurricular": "Extracurricular",
        "motivation": "Motivation",
        "internet": "Internet",
        "online courses": "OnlineCourses",
        "online course": "OnlineCourses",
        "discussions": "Discussions",
        "assignment completion": "AssignmentCompletion",
        "exam score": "ExamScore",
        "edutech": "EduTech",
        "stress level": "StressLevel",
        "stress": "StressLevel",
        "final grade": "FinalGrade"
    }

    operators = {
        "greater than or equal to": ">=",
        "at least": ">=",
        "more than or equal to": ">=",
        "less than or equal to": "<=",
        "at most": "<=",
        "more than": ">",
        "greater than": ">",
        "above": ">",
        "over": ">",
        "less than": "<",
        "below": "<",
        "under": "<",
        "equal to": "=",
        "equals": "=",
        "equal": "="
    }

    for name, column in numeric_columns.items():

        if name not in question:
            continue

        for phrase, operator in sorted(
            operators.items(),
            key=lambda x: len(x[0]),
            reverse=True
        ):

            pattern = rf'{re.escape(phrase)}\s+{re.escape(name)}\s*(?:of|is|=)?\s*(\d+(?:\.\d+)?)'

            match = re.search(pattern, question)

            if match:
                value = float(match.group(1))

                conditions.append(
                    (column, operator, value)
                )

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

    for name, column in level_columns.items():

        for level, value in level_map.items():

            patterns = [
                rf'\b{level}\s+{re.escape(name)}\b',
                rf'\b{re.escape(name)}\s+(?:is|=|of)?\s*{level}\b'
            ]

            for pattern in patterns:

                if re.search(pattern, question):
                    conditions.append(
                        (column, "=", value)
                    )
                    break

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

    result = df.copy()

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
            result = result[result[column] > value]

        elif operator == ">=":
            result = result[result[column] >= value]

        elif operator == "<":
            result = result[result[column] < value]

        elif operator == "<=":
            result = result[result[column] <= value]

        elif operator == "=":
            result = result[result[column] == value]

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

        if phrase in question:

            # Motivation is a filter when used with
            # low / medium / high motivation
            if phrase == "motivation":
                if re.search(
                    r'\b(low|medium|moderate|high)\s+motivation\b',
                    question
                ):
                    continue

            # Stress is a filter when used with
            # low / medium / high stress
            if phrase in ["stress", "stress level"]:
                if re.search(
                    r'\b(low|medium|moderate|high)\s+stress(?:\s+level)?\b',
                    question
                ):
                    continue

            column = COLUMN_MAP[phrase]

            if column not in requested:
                requested.append(column)

    return requested


def is_average_question(question):

    words = [
        "average",
        "mean",
        "avg"
    ]

    return any(
        word in question.lower()
        for word in words
    )


def is_highest_question(question):

    words = [
        "highest",
        "maximum",
        "maximum value",
        "max"
    ]

    return any(
        word in question.lower()
        for word in words
    )


def is_lowest_question(question):

    words = [
        "lowest",
        "minimum",
        "minimum value",
        "min"
    ]

    return any(
        word in question.lower()
        for word in words
    )


def is_count_question(question):

    question = question.lower()

    return (
        "how many" in question
        or "number of" in question
        or "count of" in question
    )


def calculate_answer(question, data):

    if data.empty:
        return "No students matched the given conditions."

    question_lower = question.lower()

    requested_columns = extract_requested_columns(question)

    if is_count_question(question):
        return (
            f"There are {len(data)} students "
            f"matching the given conditions."
        )

    if is_average_question(question):

        performance_columns = [
            "StudyHours",
            "Attendance",
            "Resources",
            "Extracurricular",
            "Motivation",
            "Internet",
            "OnlineCourses",
            "Discussions",
            "AssignmentCompletion",
            "ExamScore",
            "EduTech",
            "StressLevel",
            "FinalGrade"
        ]

        target_column = None

        for column in performance_columns:

            column_name = column.lower()

            if column == "StudyHours":
                keywords = ["study hours", "study hour"]

            elif column == "OnlineCourses":
                keywords = ["online courses", "online course"]

            elif column == "AssignmentCompletion":
                keywords = ["assignment completion", "assignment completion rate"]

            elif column == "ExamScore":
                keywords = ["exam score", "exam scores", "examscore"]

            elif column == "FinalGrade":
                keywords = ["final grade", "final grades", "finalgrade"]

            elif column == "StressLevel":
                keywords = ["stress level", "stress"]

            else:
                keywords = [column_name]

            for keyword in keywords:

                if keyword in question_lower:

                    is_filter = False

                    filter_patterns = [
                        rf'\b(low|medium|moderate|high)\s+{re.escape(keyword)}\b',
                        rf'\b{re.escape(keyword)}\s+(above|below|over|under|more than|less than|greater than|less than or equal to|greater than or equal to|at least|at most)',
                        rf'\b(above|below|over|under|more than|less than|greater than|less than or equal to|greater than or equal to|at least|at most)\s+{re.escape(keyword)}\b'
                    ]

                    for pattern in filter_patterns:

                        if re.search(pattern, question_lower):
                            is_filter = True
                            break

                    if not is_filter:
                        target_column = column
                        break

            if target_column:
                break

        if target_column is None:

            if requested_columns:

                for column in requested_columns:

                    if column not in ["Age", "Gender"]:

                        target_column = column
                        break

        if target_column is None:

            return (
                "Please specify what you want the average of, "
                "for example average attendance, average exam score, "
                "or average final grade."
            )

        value = data[target_column].mean()

        display_name = target_column

        display_names = {
            "StudyHours": "Study Hours",
            "Attendance": "Attendance",
            "Resources": "Resources",
            "Extracurricular": "Extracurricular",
            "Motivation": "Motivation",
            "Internet": "Internet",
            "OnlineCourses": "Online Courses",
            "Discussions": "Discussions",
            "AssignmentCompletion": "Assignment Completion",
            "ExamScore": "Exam Score",
            "EduTech": "EduTech",
            "StressLevel": "Stress Level",
            "FinalGrade": "Final Grade"
        }

        display_name = display_names.get(
            target_column,
            display_name
        )

        return (
            f"The average {display_name.lower()} "
            f"of the matching students is {value:.2f}."
        )

    if is_highest_question(question):

        if not requested_columns:
            return (
                "Please specify the column for which you want "
                "the highest value."
            )

        answers = []

        for column in requested_columns:

            if column in ["Age", "Gender"]:
                continue

            value = data[column].max()

            answers.append(
                f"Highest {column}: {value}"
            )

        if not answers:
            return (
                "Please specify a performance column such as "
                "StudyHours, Attendance, ExamScore, or FinalGrade."
            )

        return "\n".join(answers)

    if is_lowest_question(question):

        if not requested_columns:
            return (
                "Please specify the column for which you want "
                "the lowest value."
            )

        answers = []

        for column in requested_columns:

            if column in ["Age", "Gender"]:
                continue

            value = data[column].min()

            answers.append(
                f"Lowest {column}: {value}"
            )

        if not answers:
            return (
                "Please specify a performance column such as "
                "StudyHours, Attendance, ExamScore, or FinalGrade."
            )

        return "\n".join(answers)

    if requested_columns:

        columns = [
            column for column in requested_columns
            if column not in ["Age", "Gender"]
        ]

        if columns:
            return data[columns].to_string(index=False)

    return data.to_string(index=False)


def query_students(question):

    results = filter_students(question)

    print("\n==============================")
    print("STUDENT RAG QUERY")
    print("==============================")

    print("Question:", question)
    print("Matching students:", len(results))

    answer = calculate_answer(
        question,
        results
    )

    return {
        "count": len(results),
        "data": results,
        "answer": answer
    }


if __name__ == "__main__":

    while True:

        question = input(
            "\nAsk about student performance: "
        )

        if question.lower().strip() in [
            "exit",
            "quit"
        ]:
            print("Student RAG stopped.")
            break

        result = query_students(question)

        print("\nAnswer:")
        print(result["answer"])