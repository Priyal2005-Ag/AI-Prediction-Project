import os
import cv2
import pickle
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)

app = Flask(__name__)
CORS(app)


# ============================================================
# LOAD MANGO RANDOM FOREST
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "mango_random_forest.pkl"
)

with open(MODEL_PATH, "rb") as f:
    mango_rf = pickle.load(f)

print("Mango Random Forest loaded!")


# ============================================================
# MANGO CLASSES
# ============================================================

MANGO_CLASSES = {
    0: "Anthracnose",
    1: "Bacterial Black Spot",
    2: "Healthy",
    3: "Multiple"
}


# ============================================================
# FEATURE EXTRACTION
# SAME FEATURES USED DURING TRAINING
# ============================================================

def extract_mango_features(image):

    image = cv2.resize(
        image,
        (320, 320)
    )

    # --------------------------------------------------------
    # BGR COLOR FEATURES
    # --------------------------------------------------------

    b, g, r = cv2.split(image)

    color_features = []

    for channel in [b, g, r]:

        color_features.append(
            np.mean(channel)
        )

        color_features.append(
            np.std(channel)
        )


    # --------------------------------------------------------
    # HSV FEATURES
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)

    for channel in [h, s, v]:

        color_features.append(
            np.mean(channel)
        )

        color_features.append(
            np.std(channel)
        )


    # --------------------------------------------------------
    # COLOR HISTOGRAM
    # --------------------------------------------------------

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

        histogram_features.extend(
            hist
        )


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # LBP
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # GLCM
    # --------------------------------------------------------

    gray_small = (
        gray / 32
    ).astype(
        np.uint8
    )

    glcm = graycomatrix(
        gray_small,
        distances=[1],
        angles=[0],
        levels=8,
        symmetric=True,
        normed=True
    )

    glcm_features = [

        graycoprops(
            glcm,
            "contrast"
        )[0, 0],

        graycoprops(
            glcm,
            "dissimilarity"
        )[0, 0],

        graycoprops(
            glcm,
            "homogeneity"
        )[0, 0],

        graycoprops(
            glcm,
            "energy"
        )[0, 0],

        graycoprops(
            glcm,
            "correlation"
        )[0, 0]
    ]


    # --------------------------------------------------------
    # EDGE FEATURES
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_features = [

        np.mean(
            edges > 0
        ),

        np.std(
            edges
        ),

        np.sum(
            edges > 0
        )
    ]


    # --------------------------------------------------------
    # FINAL 86 FEATURES
    # --------------------------------------------------------

    features = np.concatenate([

        np.array(color_features),

        np.array(histogram_features),

        np.array(lbp_hist),

        np.array(glcm_features),

        np.array(edge_features)

    ])

    return features


# ============================================================
# MANGO PREDICTION API
# ============================================================

@app.route(
    "/predict-mango",
    methods=["POST"]
)
def predict_mango():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded"
            }), 400


        file = request.files["image"]

        image_bytes = file.read()

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if image is None:

            return jsonify({
                "error": "Invalid image"
            }), 400


        # ----------------------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------------------

        features = extract_mango_features(
            image
        )


        # ----------------------------------------------------
        # VERIFY 86 FEATURES
        # ----------------------------------------------------

        if len(features) != 86:

            return jsonify({
                "error":
                    f"Expected 86 features, got {len(features)}"
            }), 500


        features = features.reshape(
            1,
            -1
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = mango_rf.predict(
            features
        )[0]


        probabilities = mango_rf.predict_proba(
            features
        )[0]


        prediction = int(
            prediction
        )


        confidence = float(
            np.max(probabilities) * 100
        )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        probability_data = {}

        for class_id, probability in zip(
            mango_rf.classes_,
            probabilities
        ):

            probability_data[
                MANGO_CLASSES[int(class_id)]
            ] = round(
                float(probability * 100),
                2
            )


        return jsonify({

            "success": True,

            "prediction":
                MANGO_CLASSES[prediction],

            "confidence":
                round(confidence, 2),

            "probabilities":
                probability_data,

            "features":
                len(features[0])

        })


    except Exception as e:

        print(
            "Mango prediction error:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )