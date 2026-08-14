import os
import json
import pickle
import time
import numpy as np
import cv2
import streamlit as st

from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🤖",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CNN_PATH = os.path.join(
    BASE_DIR,
    "lung_cancer_dnn_model.keras"
)

MANGO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "fruit disease classification",
    "mango_random_forest.pkl"
)

KMEANS_PATH = os.path.join(
    BASE_DIR,
    "student_kmeans.json"
)

CHROMA_PATH = os.path.join(
    BASE_DIR,
    "RAG",
    "chroma_db"
)


CNN_CLASSES = {
    0: "Benign",
    1: "Malignant",
    2: "Normal"
}

MANGO_CLASSES = {
    0: "Anthracnose",
    1: "Bacterial Black Spot",
    2: "Healthy",
    3: "Multiple"
}


if "cnn_result" not in st.session_state:
    st.session_state.cnn_result = None

if "mango_result" not in st.session_state:
    st.session_state.mango_result = None

if "kmeans_result" not in st.session_state:
    st.session_state.kmeans_result = None

if "rag_result" not in st.session_state:
    st.session_state.rag_result = None


@st.cache_resource
def load_cnn():
    import tensorflow as tf
    
    return tf.keras.models.load_model(CNN_PATH)


@st.cache_resource
def load_mango_model():
    with open(MANGO_MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_kmeans():
    with open(KMEANS_PATH, "r") as f:
        return json.load(f)


@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return db


def extract_mango_features(image):

    image = cv2.resize(image, (320, 320))

    b, g, r = cv2.split(image)

    color_features = []

    for channel in [b, g, r]:
        color_features.append(np.mean(channel))
        color_features.append(np.std(channel))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    for channel in [h, s, v]:
        color_features.append(np.mean(channel))
        color_features.append(np.std(channel))

    histogram_features = []

    for channel in [b, g, r]:

        hist = cv2.calcHist(
            [channel],
            [0],
            None,
            [16],
            [0, 256]
        )

        hist = cv2.normalize(
            hist,
            hist
        ).flatten()

        histogram_features.extend(hist)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    radius = 2
    points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        points,
        radius,
        method="uniform"
    )

    n_bins = points + 2

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )

    lbp_hist = lbp_hist.astype(float)

    lbp_hist /= (
        lbp_hist.sum() + 1e-7
    )

    gray_small = (
        gray / 32
    ).astype(np.uint8)

    glcm = graycomatrix(
        gray_small,
        distances=[1],
        angles=[0],
        levels=8,
        symmetric=True,
        normed=True
    )

    glcm_features = [
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "dissimilarity")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        graycoprops(glcm, "correlation")[0, 0]
    ]

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_features = [
        np.mean(edges > 0),
        np.std(edges),
        np.sum(edges > 0)
    ]

    features = np.concatenate([
        np.array(color_features),
        np.array(histogram_features),
        np.array(lbp_hist),
        np.array(glcm_features),
        np.array(edge_features)
    ])

    return features


def predict_cnn(image):

    image = image.resize((320, 320))

    image_array = np.array(image)

    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    image_array = image_array.astype(np.float32) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    model = load_cnn()

    prediction = model.predict(
        image_array,
        verbose=0
    )

    probabilities = prediction[0]

    index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[index] * 100
    )

    return (
        CNN_CLASSES[index],
        confidence
    )


def predict_mango(image):

    image_array = np.array(image)

    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    image_array = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    features = extract_mango_features(
        image_array
    )

    if len(features) != 86:
        raise ValueError(
            f"Expected 86 features, got {len(features)}"
        )

    features = features.reshape(
        1,
        -1
    )

    model = load_mango_model()

    prediction = int(
        model.predict(features)[0]
    )

    probabilities = model.predict_proba(
        features
    )[0]

    confidence = float(
        np.max(probabilities) * 100
    )

    return (
        MANGO_CLASSES[prediction],
        confidence
    )


def predict_kmeans(values):

    model = load_kmeans()

    means = (
        model.get("scaler_mean")
        if model.get("scaler_mean") is not None
        else model.get("mean")
    )

    scales = (
        model.get("scaler_scale")
        if model.get("scaler_scale") is not None
        else model.get("scale")
    )

    centers = model.get(
        "cluster_centers"
    )

    labels = model.get(
        "cluster_labels",
        {}
    )

    if means is None:
        raise ValueError(
            "KMeans scaler mean not found."
        )

    if scales is None:
        raise ValueError(
            "KMeans scaler scale not found."
        )

    if centers is None:
        raise ValueError(
            "KMeans cluster centers not found."
        )

    scaled = []

    for i, value in enumerate(values):

        scale = scales[i]

        if scale == 0:
            scale = 1

        scaled.append(
            (value - means[i]) / scale
        )

    scaled = np.array(
        scaled,
        dtype=float
    )

    distances = []

    for center in centers:

        center = np.array(
            center,
            dtype=float
        )

        distance = np.sqrt(
            np.sum(
                (
                    scaled -
                    center[:len(scaled)]
                ) ** 2
            )
        )

        distances.append(distance)

    cluster = int(
        np.argmin(distances)
    )

    label = labels.get(
        str(cluster),
        labels.get(
            cluster,
            "Unknown"
        )
    )

    return (
        cluster,
        label,
        distances
    )


def ask_rag(question):

    db = load_rag()

    documents = db.similarity_search(
        question,
        k=5
    )

    if not documents:
        return (
            "No relevant information was found.",
            []
        )

    context_parts = []
    sources = []

    for document in documents:

        context_parts.append(
            document.page_content
        )

        source = document.metadata.get(
            "source",
            "Knowledge Base"
        )

        if source not in sources:
            sources.append(source)

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are an AI assistant for an AI Disease Prediction System.

Answer the user's question using only the provided knowledge base context.

If the answer is not available in the context, clearly say that the information is not available in the knowledge base.

Do not invent medical facts.

Knowledge Base Context:

{context}

User Question:

{question}

Answer:
"""

    llm = OllamaLLM(
    model="llama3.2:latest",
    base_url="https://put-lou-ottawa-switched.trycloudflare.com "
    )

    answer = llm.invoke(
        prompt
    )

    return answer, sources

st.sidebar.title("AI Models")

selected_model = st.sidebar.radio(
    "Select Model",
    [
        "CNN",
        "Random Forest",
        "KMeans",
        "About Project"
    ]
)


if selected_model == "CNN":

    st.title("🫁 Lung Cancer Prediction")

    st.write(
        "CNN-based lung cancer image classification."
    )

    uploaded_file = st.file_uploader(
    "Upload Lung Cancer Image",
    type=["jpg", "jpeg", "png"],
    key="cnn_upload"
    )

    if uploaded_file:

      current_file_id = (uploaded_file.name,
        uploaded_file.size
        )

    if st.session_state.get("cnn_file_id") != current_file_id:
        st.session_state.cnn_file_id = current_file_id
        st.session_state.cnn_result = None


        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=320
        )

        if st.button(
            "Predict Lung Cancer",
            key="cnn_predict"
        ):

            with st.spinner(
                "Running CNN prediction..."
            ):

                try:

                    start = time.perf_counter()

                    prediction, confidence = predict_cnn(
                        image
                    )

                    end = time.perf_counter()

                    st.session_state.cnn_result = {
                        "prediction": prediction,
                        "confidence": confidence,
                        "time": (end - start) * 1000
                    }

                except Exception as e:

                    st.error(
                        f"CNN Error: {e}"
                    )

        if st.session_state.cnn_result:

            result = st.session_state.cnn_result

            st.success(
                f"Prediction: {result['prediction']}"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Confidence",
                    f"{result['confidence']:.2f}%"
                )

            with col2:
                st.metric(
                    "Prediction Time",
                    f"{result['time']:.2f} ms"
                )

    st.divider()

    st.subheader("🤖 AI RAG Assistant")

    question = st.text_area(
        "Ask a question about Lung Cancer",
        height=120,
        placeholder="Ask your question here...",
        key="cnn_rag_question"
    )

    if st.button(
        "Ask AI",
        key="cnn_ask_ai"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "AI is thinking..."
            ):

                try:

                    answer, sources = ask_rag(
                        question
                    )

                    st.session_state.rag_result = {
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    }

                except Exception as e:

                    st.error(
                        f"RAG Error: {e}"
                    )

    if st.session_state.rag_result:

        result = st.session_state.rag_result

        st.subheader("AI Answer")

        st.write(
            result["answer"]
        )

        if result["sources"]:

            st.subheader("Sources")

            for source in result["sources"]:
                st.write(
                    f"• {source}"
                )


elif selected_model == "Random Forest":

    st.title(
        "🌿 Mango Disease Classification"
    )

    st.write(
        "Random Forest-based mango disease classification."
    )

    uploaded_file = st.file_uploader(
    "Upload Mango Leaf Image",
    type=["jpg", "jpeg", "png"],
    key="mango_upload"
    )

    if uploaded_file:

      current_file_id = (
        uploaded_file.name,
        uploaded_file.size
    )

    if st.session_state.get("mango_file_id") != current_file_id:
        st.session_state.mango_file_id = current_file_id
        st.session_state.mango_result = None


        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Mango Image",
            width=320
        )

        if st.button(
            "Predict Mango Disease",
            key="mango_predict"
        ):

            with st.spinner(
                "Processing mango image..."
            ):

                try:

                    start = time.perf_counter()

                    prediction, confidence = predict_mango(
                        image
                    )

                    end = time.perf_counter()

                    st.session_state.mango_result = {
                        "prediction": prediction,
                        "confidence": confidence,
                        "time": (end - start) * 1000
                    }

                except Exception as e:

                    st.error(
                        f"Random Forest Error: {e}"
                    )

        if st.session_state.mango_result:

            result = st.session_state.mango_result

            st.success(
                f"Prediction: {result['prediction']}"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Confidence",
                    f"{result['confidence']:.2f}%"
                )

            with col2:
                st.metric(
                    "Prediction Time",
                    f"{result['time']:.2f} ms"
                )

    st.divider()

    st.subheader("🤖 AI RAG Assistant")

    question = st.text_area(
        "Ask a question about Mango Diseases",
        height=120,
        placeholder="Ask your question here...",
        key="mango_rag_question"
    )

    if st.button(
        "Ask AI",
        key="mango_ask_ai"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "AI is thinking..."
            ):

                try:

                    answer, sources = ask_rag(
                        question
                    )

                    st.session_state.rag_result = {
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    }

                except Exception as e:

                    st.error(
                        f"RAG Error: {e}"
                    )

    if st.session_state.rag_result:

        result = st.session_state.rag_result

        st.subheader("AI Answer")

        st.write(
            result["answer"]
        )

        if result["sources"]:

            st.subheader("Sources")

            for source in result["sources"]:
                st.write(
                    f"• {source}"
                )


elif selected_model == "KMeans":

    st.title(
        "🎓 Student Performance Clustering"
    )

    st.write(
        "KMeans clustering for student performance analysis."
    )

    col1, col2 = st.columns(2)

    with col1:

        study_hours = st.number_input(
            "Study Hours",
            min_value=0.0,
            key="study_hours"
        )

        attendance = st.number_input(
            "Attendance",
            min_value=0.0,
            key="attendance"
        )

        motivation = st.number_input(
            "Motivation",
            min_value=0.0,
            key="motivation"
        )

        online_courses = st.number_input(
            "Online Courses",
            min_value=0.0,
            key="online_courses"
        )

        discussions = st.number_input(
            "Discussions",
            min_value=0.0,
            key="discussions"
        )

        assignment_completion = st.number_input(
            "Assignment Completion",
            min_value=0.0,
            key="assignment_completion"
        )

    with col2:

        exam_score = st.number_input(
            "Exam Score",
            min_value=0.0,
            key="exam_score"
        )

        stress_level = st.number_input(
            "Stress Level",
            min_value=0.0,
            key="stress_level"
        )

        final_grade = st.number_input(
            "Final Grade",
            min_value=0.0,
            key="final_grade"
        )

        age = st.number_input(
            "Age",
            min_value=0.0,
            key="age"
        )

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male"
            ],
            key="gender"
        )

    if st.button(
        "Predict Student Cluster",
        key="kmeans_predict"
    ):

        try:

            gender_value = (
                0
                if gender == "Female"
                else 1
            )

            values = [
                study_hours,
                attendance,
                motivation,
                online_courses,
                discussions,
                assignment_completion,
                exam_score,
                stress_level,
                final_grade,
                age,
                gender_value
            ]

            start = time.perf_counter()

            cluster, label, distances = predict_kmeans(
                values
            )

            end = time.perf_counter()

            st.session_state.kmeans_result = {
                "cluster": cluster,
                "label": label,
                "time": (end - start) * 1000,
                "distances": distances
            }

        except Exception as e:

            st.error(
                f"KMeans Error: {e}"
            )

    if st.session_state.kmeans_result:

        result = st.session_state.kmeans_result

        st.success(
            f"Cluster: {result['cluster']}"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Performance",
                str(result["label"])
            )

        with col2:

            st.metric(
                "Prediction Time",
                f"{result['time']:.2f} ms"
            )

        st.subheader(
            "Cluster Distances"
        )

        for i, distance in enumerate(
            result["distances"]
        ):

            st.write(
                f"Cluster {i}: {distance:.4f}"
            )

    st.divider()

    st.subheader("🤖 AI RAG Assistant")

    question = st.text_area(
        "Ask a question about Student Performance",
        height=120,
        placeholder="Ask your question here...",
        key="kmeans_rag_question"
    )

    if st.button(
        "Ask AI",
        key="kmeans_ask_ai"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "AI is thinking..."
            ):

                try:

                    answer, sources = ask_rag(
                        question
                    )

                    st.session_state.rag_result = {
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    }

                except Exception as e:

                    st.error(
                        f"RAG Error: {e}"
                    )

    if st.session_state.rag_result:

        result = st.session_state.rag_result

        st.subheader("AI Answer")

        st.write(
            result["answer"]
        )

        if result["sources"]:

            st.subheader("Sources")

            for source in result["sources"]:
                st.write(
                    f"• {source}"
                )


else:

    st.title(
        "📘 About Project"
    )

    st.write(
        "This is an AI-based prediction system combining Machine Learning, Deep Learning, clustering, and Retrieval-Augmented Generation."
    )

    st.subheader(
        "CNN"
    )

    st.write(
        "Used for lung cancer image classification with three classes: Benign, Malignant, and Normal."
    )

    st.subheader(
        "Random Forest"
    )

    st.write(
        "Used for supervised mango disease classification with four classes: Anthracnose, Bacterial Black Spot, Healthy, and Multiple."
    )

    st.subheader(
        "KMeans"
    )

    st.write(
        "Used for student performance clustering."
    )

    st.subheader(
        "RAG"
    )

    st.write(
        "Retrieves relevant information from the knowledge base and generates an answer using a local language model."
    )
