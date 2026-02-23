import os
import numpy as np
from tensorflow.keras.models import load_model
from inference.preprocess import extract_faces

# ==============================
# CONFIGURATION (MUST MATCH BACKEND)
# ==============================
IMG_SIZE = 128
NUM_SAMPLES = 10
THRESHOLD = 0.40

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model",
    "cnn_model.h5"
)

# ==============================
# LOAD MODEL
# ==============================
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    exit(1)

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_video(video_path):

    if not os.path.exists(video_path):
        return "No face detected", 0.0, {}

    faces = extract_faces(video_path)

    if len(faces) == 0:
        return "No face detected", 0.0, {}

    # ==============================
    # SAVE DISPLAY COPY BEFORE MODEL USE
    # faces are in [-1, 1] because of preprocess_input
    # convert back to [0, 255] for display
    # ==============================
    display_frames = ((faces + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    # ==============================
    # MODEL PREDICTION (UNCHANGED)
    # ==============================
    predictions = model.predict(faces, verbose=0).flatten()

    # ==============================
    # FRAME LEVEL DECISION (SAME AS BACKEND)
    # ==============================
    binary_preds = (predictions > THRESHOLD).astype(int)

    fake_count = int(np.sum(binary_preds))
    real_count = len(binary_preds) - fake_count
    total_frames = len(predictions)

    fake_percentage = (fake_count / total_frames) * 100
    real_percentage = (real_count / total_frames) * 100

    # ==============================
    # FINAL DECISION (MEAN SCORE — SAME)
    # ==============================
    final_score = float(np.mean(predictions))

    if final_score > THRESHOLD:
        label = "FAKE"
        confidence = fake_percentage / 100.0
    else:
        label = "REAL"
        confidence = real_percentage / 100.0

    # ==============================
    # RETURN SAFE DATA FOR STREAMLIT
    # ==============================
    return label, confidence, {
        "mean_score": final_score,
        "threshold": THRESHOLD,
        "fake_frames": fake_count,
        "real_frames": real_count,
        "total_frames": total_frames,
        "fake_percentage": fake_percentage,
        "real_percentage": real_percentage,
        "frames": display_frames.tolist()   # SAFE FOR UI
    }