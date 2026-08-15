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

    operator_words = {
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

    for column_name, column in sorted(
        numeric_columns.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):
        for phrase, operator in operator_words.items():

            pattern_a = (
                rf'\b{re.escape(column_name)}\b\s+'
                rf'{re.escape(phrase)}\s+'
                rf'(\d+(?:\.\d+)?)\b'
            )

            for match in re.finditer(pattern_a, question):
                conditions.append(
                    (column, operator, float(match.group(1)))
                )

            pattern_b = (
                rf'\b{re.escape(phrase)}\s+'
                rf'(\d+(?:\.\d+)?)\s+'
                rf'{re.escape(column_name)}\b'
            )

            for match in re.finditer(pattern_b, question):
                conditions.append(
                    (column, operator, float(match.group(1)))
                )

            pattern_c = (
                rf'\b(\d+(?:\.\d+)?)\s+'
                rf'{re.escape(phrase)}\s+'
                rf'{re.escape(column_name)}\b'
            )

            reverse_operator = {
                ">": "<",
                "<": ">",
                ">=": "<=",
                "<=": ">=",
                "=": "="
            }

            for match in re.finditer(pattern_c, question):
                conditions.append(
                    (
                        column,
                        reverse_operator[operator],
                        float(match.group(1))
                    )
                )

    symbol_pattern = re.compile(
        r'(?:(?P<column1>[a-zA-Z ]+?)\s*'
        r'(?P<operator1>>=|<=|>|<|=)\s*'
        r'(?P<value1>\d+(?:\.\d+)?)'
        r'|'
        r'(?P<value2>\d+(?:\.\d+)?)\s*'
        r'(?P<operator2>>=|<=|>|<|=)\s*'
        r'(?P<column2>[a-zA-Z ]+?))(?=\s*(?:and|or|,|$))'
    )

    for match in symbol_pattern.finditer(question):

        if match.group("column1") and match.group("value1"):
            column_text = match.group("column1").strip()

            for name, column in sorted(
                numeric_columns.items(),
                key=lambda x: len(x[0]),
                reverse=True
            ):
                if name in column_text:
                    conditions.append(
                        (
                            column,
                            match.group("operator1"),
                            float(match.group("value1"))
                        )
                    )
                    break

        elif match.group("column2") and match.group("value2"):
            column_text = match.group("column2").strip()

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
                if name in column_text:
                    conditions.append(
                        (
                            column,
                            reverse_operator[match.group("operator2")],
                            float(match.group("value2"))
                        )
                    )
                    break

    unique_conditions = []

    for condition in conditions:
        if condition not in unique_conditions:
            unique_conditions.append(condition)

    return unique_conditions


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

        column_priority = [
            ("study hours", "StudyHours"),
            ("study hour", "StudyHours"),
            ("attendance", "Attendance"),
            ("resources", "Resources"),
            ("extracurricular", "Extracurricular"),
            ("motivation", "Motivation"),
            ("internet", "Internet"),
            ("online courses", "OnlineCourses"),
            ("online course", "OnlineCourses"),
            ("discussions", "Discussions"),
            ("assignment completion rate", "AssignmentCompletion"),
            ("assignment completion", "AssignmentCompletion"),
            ("exam scores", "ExamScore"),
            ("exam score", "ExamScore"),
            ("examscore", "ExamScore"),
            ("edutech", "EduTech"),
            ("stress level", "StressLevel"),
            ("stress", "StressLevel"),
            ("final grades", "FinalGrade"),
            ("final grade", "FinalGrade")
        ]

        for phrase, column in column_priority:

            if phrase not in question_lower:
                continue

            filter_patterns = [
                rf'\b{re.escape(phrase)}\s+'
                rf'(above|below|over|under|greater than|less than|'
                rf'more than|at least|at most|equal to|equals|equal)'
                rf'\s+\d+',

                rf'\b(above|below|over|under|greater than|less than|'
                rf'more than|at least|at most|equal to|equals|equal)'
                rf'\s+\d+\s+{re.escape(phrase)}',

                rf'\b{re.escape(phrase)}\s*'
                rf'(>=|<=|>|<|=)\s*\d+',

                rf'\b\d+\s*(>=|<=|>|<|=)\s*'
                rf'{re.escape(phrase)}'
            ]

            is_filter = any(
                re.search(pattern, question_lower)
                for pattern in filter_patterns
            )

            if not is_filter:
                target_column = column
                break

        if target_column is None:

            for column in requested_columns:
                if column in performance_columns:
                    target_column = column
                    break

        if target_column is None:
            return (
                "Please specify what you want the average of, "
                "for example average attendance, "
                "average exam score, or average final grade."
            )

        value = data[target_column].mean()

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
            target_column
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