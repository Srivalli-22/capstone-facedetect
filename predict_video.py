import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
import os

# ==============================
# CONFIGURATION (EDIT HERE)
# ==============================
MODEL_PATH = "model/cnn_model.h5"
VIDEO_PATH = "test.mp4"

IMG_SIZE = 128         # CHANGE ANYTIME (64, 96, 128, 224...)
NUM_SAMPLES = 10       # Number of frames to sample
THRESHOLD = 0.40       # Final decision threshold

TEMPORAL_SAMPLING = 5  # Keep same as training

# ==============================
# LOAD MODEL
# ==============================
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    exit(1)

# ==============================
# LOAD FACE DETECTOR
# ==============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==============================
# FACE EXTRACTION FUNCTION
# ==============================
def extract_faces(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        faces = []
        frame_count = 0

        if not cap.isOpened():
            print("❌ Could not open video.")
            return np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype="float32")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % TEMPORAL_SAMPLING == 0:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    if len(detected_faces) > 0:
                        (x, y, w, h) = max(detected_faces, key=lambda f: f[2] * f[3])
                        face = frame[y:y+h, x:x+w]
                    else:
                        # Center crop fallback
                        h, w = frame.shape[:2]
                        crop = min(h, w)
                        start_h = (h - crop) // 2
                        start_w = (w - crop) // 2
                        face = frame[start_h:start_h+crop, start_w:start_w+crop]

                    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                    faces.append(face)

                except Exception:
                    pass  # Skip problematic frames safely

            frame_count += 1

            if len(faces) >= NUM_SAMPLES:
                break

        cap.release()

        if len(faces) == 0:
            return np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype="float32")

        faces = np.array(faces, dtype="float32")

        # Normalize
        faces = faces / 255.0

        # Xception preprocessing
        faces = preprocess_input(faces)

        return faces

    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        return np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype="float32")


# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_video(video_path):

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    print(f"\n🎬 Processing: {video_path}")

    faces = extract_faces(video_path)

    if len(faces) == 0:
        print("❌ No valid frames extracted.")
        return

    print(f"📊 Extracted {len(faces)} frames")

    try:
        predictions = model.predict(faces, verbose=0).flatten()
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return

    # ==============================
    # FRAME LEVEL DECISION
    # ==============================
    binary_preds = (predictions > THRESHOLD).astype(int)

    fake_count = np.sum(binary_preds)
    real_count = len(binary_preds) - fake_count

    fake_percentage = (fake_count / len(predictions)) * 100
    real_percentage = (real_count / len(predictions)) * 100

    # ==============================
    # FINAL DECISION (Mean score)
    # ==============================
    final_score = np.mean(predictions)

    if final_score > THRESHOLD:
        label = "FAKE"
        confidence = fake_percentage
    else:
        label = "REAL"
        confidence = real_percentage

    # ==============================
    # PRINT RESULTS
    # ==============================
    print("\n✅ FINAL PREDICTION")
    print("=" * 50)
    print(f"Result         : {label}")
    print(f"Confidence     : {confidence:.2f}%")
    print("-" * 50)
    print(f"Mean Score     : {final_score:.4f}")
    print(f"Threshold      : {THRESHOLD}")
    print(f"Fake Frames    : {fake_count}/{len(predictions)} ({fake_percentage:.1f}%)")
    print(f"Real Frames    : {real_count}/{len(predictions)} ({real_percentage:.1f}%)")
    print("=" * 50)


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    predict_video(VIDEO_PATH)