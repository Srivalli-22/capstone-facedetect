import cv2
import numpy as np
from tensorflow.keras.applications.xception import preprocess_input

# ---------------- CONFIG ----------------
IMG_SIZE = 128
FRAMES_PER_VIDEO = 10
TEMPORAL_SAMPLING = 5
# ----------------------------------------

# Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_faces(video_path):
    """
    Extract faces from video.
    MUST match training pipeline exactly.
    """

    try:
        cap = cv2.VideoCapture(video_path)
        faces = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Temporal sampling
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
                        crop_size = min(h, w)
                        start_h = (h - crop_size) // 2
                        start_w = (w - crop_size) // 2
                        face = frame[start_h:start_h+crop_size,
                                     start_w:start_w+crop_size]

                    # Resize
                    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE),
                                      interpolation=cv2.INTER_LINEAR)

                    # Convert BGR → RGB
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                    faces.append(face)

                except Exception as e:
                    print(f"Warning processing frame: {str(e)}")
                    continue

            frame_count += 1

            if len(faces) >= FRAMES_PER_VIDEO:
                break

        cap.release()

        if len(faces) == 0:
            return np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype="float32")

        # Convert to float32
        faces = np.array(faces, dtype="float32")

        # IMPORTANT: DO NOT divide by 255
        # Xception preprocess_input expects range [0,255]
        faces = preprocess_input(faces)

        return faces

    except Exception as e:
        print(f"Error reading video {video_path}: {str(e)}")
        return np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype="float32")