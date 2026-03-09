import tempfile
import os
import sys

# allow backend to access project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.predict import predict_video

def analyze_video(uploaded_file):

    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    # Run prediction
    label, confidence, details = predict_video(video_path)

    # Remove temp file
    os.remove(video_path)

    return label, confidence, details