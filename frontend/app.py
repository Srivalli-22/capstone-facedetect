import streamlit as st
import tempfile
import os
import sys
import numpy as np

# Allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference.predict import predict_video

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DeepFake Detection System",
    page_icon="🕵️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    font-family: 'Segoe UI', sans-serif;
    color: #ffffff;
}

/* Layout Padding */
.block-container {
    padding-left: 6rem;
    padding-right: 6rem;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.08);
    padding: 30px;
    border-radius: 16px;
    margin-top: 30px;
    backdrop-filter: blur(12px);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.4);
}

/* Section Titles */
.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #00e5ff;
    margin-bottom: 20px;
}

/* Header */
.main-title {
    text-align: center;
    font-size: 64px;
    font-weight: 800;
    color: #4fc3f7;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #d9f1ff;
    max-width: 850px;
    margin: auto;
    line-height: 1.7;
    margin-bottom: 40px;
}

/* Uploaded file highlight */
.file-box {
    background: rgba(0,229,255,0.15);
    padding: 12px 15px;
    border-radius: 8px;
    color: #00e5ff;
    font-weight: 600;
    margin-top: 10px;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 18px;
    font-weight: 600;
    border-radius: 10px;
    padding: 12px;
    border: none;
}

.stButton > button:hover {
    transform: scale(1.03);
}

/* Result Box */
.result-box {
    padding: 40px;
    border-radius: 14px;
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 25px;
}

.real {
    background: linear-gradient(90deg, #00b09b, #96c93d);
}

.fake {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
}

/* Confidence text */
.confidence-text {
    font-size: 20px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 10px;
}

/* 🔥 FINAL STRONG CHECKBOX FIX */
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label p,
[data-testid="stCheckbox"] span,
[data-testid="stCheckbox"] div {
    color: #ffffff !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* Remove any faded styling */
[data-testid="stCheckbox"] * {
    opacity: 1 !important;
}

/* Technical Details Panel */
.details-box {
    background: rgba(255,255,255,0.07);
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
    color: #ffffff;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 60px;
    color: #b3e5fc;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>Advanced AI DeepFake Detection</div>", unsafe_allow_html=True)

st.markdown("""
<div class='subtitle'>
Protect yourself from manipulated and AI-generated videos using our GAN-powered deep learning framework.
Our system analyzes facial inconsistencies, frame-level artifacts, and spatio-temporal patterns
to accurately detect whether a video is REAL or FAKE in seconds.
</div>
""", unsafe_allow_html=True)

# ---------------- HOW IT WORKS ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📌 How It Works</div>', unsafe_allow_html=True)

st.markdown("""
1️⃣ Upload a video containing a human face.  
2️⃣ Facial frames are extracted and preprocessed.  
3️⃣ A trained Xception CNN analyzes the frames.  
4️⃣ The system predicts **REAL** or **FAKE** with confidence.
""")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- UPLOAD SECTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📤 Upload Video File</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Supported formats: MP4, AVI, MOV",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:
    st.markdown(
        f"<div class='file-box'>📁 Uploaded File: {uploaded_file.name}</div>",
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ANALYZE ----------------
if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    if st.button("🔍 Analyze Video", use_container_width=True):

        with st.spinner("Analyzing video frames..."):
            label, confidence, details = predict_video(video_path)

        st.session_state["label"] = label
        st.session_state["confidence"] = confidence
        st.session_state["details"] = details

    os.remove(video_path)

# ---------------- RESULT ----------------
if "label" in st.session_state:

    label = st.session_state["label"]
    confidence = st.session_state["confidence"]
    details = st.session_state["details"]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Detection Result</div>', unsafe_allow_html=True)

    if label == "FAKE":
        st.markdown("<div class='result-box fake'>🚨 FAKE VIDEO</div>", unsafe_allow_html=True)
    elif label == "REAL":
        st.markdown("<div class='result-box real'>✅ REAL VIDEO</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No face detected")

    st.markdown(f"<div class='confidence-text'>Confidence Score: {confidence * 100:.2f}%</div>", unsafe_allow_html=True)
    st.progress(min(int(confidence * 100), 100))

    if st.checkbox("Show Technical Details"):
        if details:
            st.markdown('<div class="details-box">', unsafe_allow_html=True)
            st.write("Mean Score:", round(details.get("mean_score", 0), 4))
            st.write("Threshold:", details.get("threshold", 0.40))
            st.write("Fake Frames:", f"{details.get('fake_frames', 0)}/{details.get('total_frames', 0)}")
            st.write("Real Frames:", f"{details.get('real_frames', 0)}/{details.get('total_frames', 0)}")
            st.markdown('</div>', unsafe_allow_html=True)

    if st.checkbox("Show Extracted Frames"):
        frames = details.get("frames", [])
        if frames:
            st.markdown("### 🖼 Extracted Frames")
            cols = st.columns(6)
            for i, frame in enumerate(frames):
                frame_array = np.array(frame, dtype=np.uint8)
                cols[i % 6].image(frame_array, width=100)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class='footer'>
DeepFake Detection System | Built with TensorFlow, OpenCV & Streamlit
</div>
""", unsafe_allow_html=True)