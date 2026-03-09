import os
import cv2
import numpy as np
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# ----------------------------
# CONFIG
# ----------------------------
DATASET_PATH = r"D:\deepfake\dataset"
REAL_DIR = "real train"
FAKE_DIR = "fake train"
IMG_SIZE = 128
FRAMES_PER_VIDEO = 10   # Reduced for stability
BATCH_SIZE = 8
EPOCHS = 5
TEMPORAL_SAMPLING = 5   # Every 5th frame for consistency

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ----------------------------
# FRAME EXTRACTION + PREPROCESSING
# ----------------------------
def extract_faces_from_video(video_path, label):
    """
    Extract faces from video with:
    - 128x128 resize
    - Proper normalization (0-1)
    - Xception preprocessing
    - Temporal sampling (every 5th frame)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Temporal sampling
            if frame_count % TEMPORAL_SAMPLING == 0:
                try:
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    if len(faces) > 0:
                        # Get the largest face
                        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                        face = frame[y:y+h, x:x+w]
                    else:
                        # Use center crop as fallback
                        h, w = frame.shape[:2]
                        crop_size = min(h, w)
                        start_h = (h - crop_size) // 2
                        start_w = (w - crop_size) // 2
                        face = frame[start_h:start_h+crop_size, start_w:start_w+crop_size]

                    # Resize to 128x128
                    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                    
                    # Convert BGR to RGB
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    
                    frames.append(face)

                except Exception as e:
                    print(f"Error processing frame in {video_path}: {str(e)}")
                    continue

            frame_count += 1
            if len(frames) >= FRAMES_PER_VIDEO:
                break

        cap.release()
        return frames

    except Exception as e:
        print(f"Error reading video {video_path}: {str(e)}")
        return []

# ----------------------------
# LOAD DATASET
# ----------------------------
X = []
y = []

print("Loading dataset...")
for label, folder in enumerate([REAL_DIR, FAKE_DIR]):
    folder_path = os.path.join(DATASET_PATH, folder)
    video_count = 0
    
    if not os.path.exists(folder_path):
        print(f"⚠️  Folder not found: {folder_path}")
        continue
    
    for video in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video)
        data = extract_faces_from_video(video_path, label)

        if len(data) > 0:
            for face in data:
                X.append(face)
                y.append(label)
            video_count += 1
            print(f"  ✓ {folder}: {video} ({len(data)} frames)")

    print(f"Total videos processed from {folder}: {video_count}")

X = np.array(X, dtype="float32")
y = np.array(y)

print(f"\n📊 Dataset Summary:")
print(f"  Total frames: {len(X)}")
print(f"  Real frames: {np.sum(y == 0)}")
print(f"  Fake frames: {np.sum(y == 1)}")

# ----------------------------
# NORMALIZE TO 0-1 RANGE
# ----------------------------
X = X / 255.0

# ----------------------------
# APPLY XCEPTION PREPROCESSING
# ----------------------------
X = preprocess_input(X)

# ----------------------------
# TRAIN / TEST SPLIT
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# MODEL (XCEPTION)
# ----------------------------
base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Fine-tune: freeze most layers, unfreeze top X layers
for layer in base_model.layers[:-36]:
    layer.trainable = False
for layer in base_model.layers[-36:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile with a low learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Data augmentation for robustness
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.05,
    zoom_range=0.08,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Compute class weights to handle any imbalance
from sklearn.utils import class_weight
classes = np.unique(y_train)
class_weights_array = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = {int(c): w for c, w in zip(classes, class_weights_array)}

# Callbacks
model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "deepfake_xception_model.h5")

callbacks = [
    ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

print("\n🚀 Starting training with augmentation and fine-tuning...")

train_gen = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

print(f"\n✅ Training complete. Best model saved to: {model_path}")

# Print final metrics
train_acc = history.history.get('accuracy', [None])[-1]
val_acc = history.history.get('val_accuracy', [None])[-1]
if train_acc is not None and val_acc is not None:
    print(f"\n📈 Final Training Accuracy: {train_acc*100:.2f}%")
    print(f"📈 Final Validation Accuracy: {val_acc*100:.2f}%")
