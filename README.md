DeepFake Video Detection System

An AI-powered DeepFake Detection system that analyzes uploaded videos and determines whether they are REAL or FAKE using a deep learning model based on the Xception architecture.
The system extracts facial frames from videos, processes them using a trained CNN model, and displays the prediction through a Streamlit-based user interface.

Project Overview

DeepFake technology can manipulate videos by replacing or altering faces using AI. While useful in entertainment, it also poses serious risks such as misinformation, identity misuse, and fraud.

This project detects DeepFake videos by:

Extracting facial frames from the uploaded video

Preprocessing the frames for the neural network

Running the frames through a trained Xception CNN model

Aggregating frame predictions

Displaying the final result with a confidence score

The system provides a user-friendly interface where users can upload videos and instantly see detection results.

Features

Upload video files for analysis

Automatic face extraction from frames

Deep learning-based DeepFake detection

Frame-level prediction analysis

Confidence score visualization

Option to view extracted frames

Technical prediction details

Clean and interactive Streamlit interface

Project Architecture
User Uploads Video
        │
        ▼
Frontend (Streamlit UI)
frontend/main.py
        │
        ▼
Backend Processing
backend/app.py
        │
        ▼
Prediction Engine
backend/predict.py
        │
        ▼
Face Extraction
backend/preprocess.py
        │
        ▼
Deep Learning Model
cnn_model.h5
Project Structure
DEEPFAKE
│
├── backend
│   ├── __init__.py
│   ├── app.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train_xception.py
│
├── frontend
│   └── main.py
│
├── dataset
│   ├── REAL
│   └── FAKE
│
├── Predict_videos
│
├── haarcascade_frontalface_default.xml
├── predict_video.py
└── .gitignore
Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Streamlit

Scikit-learn

Deep Learning Model

The detection model is built using the Xception Convolutional Neural Network, which is highly effective for image classification tasks.

Key details:

Architecture: Xception

Input size: 128 × 128

Frame sampling from videos

Binary classification (REAL vs FAKE)

Sigmoid activation output

Mean prediction score used for final decision

Dataset

The dataset contains two categories:

dataset/
   REAL/
   FAKE/

Each folder contains videos used to train the model.

During training:

Faces are extracted from frames

Frames are normalized and preprocessed

Data augmentation is applied

Model is trained using transfer learning

Model File

The trained model file cnn_model.h5 is not included in this repository because it exceeds GitHub's file size limit.

To generate the model, run:

python backend/train_xception.py

