import cv2
import json
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model


# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "hybrid_emotion_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "models", "hybrid_labels.json")


# =========================
# LOAD MODEL AND LABELS
# =========================

print("Loading hybrid model...")
model = load_model(MODEL_PATH)
print("Hybrid model loaded successfully.")

with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)

# Convert {'angry':0, 'happy':1} to {0:'angry', 1:'happy'}
idx_to_class = {v: k for k, v in class_indices.items()}

print("Classes:", idx_to_class)


# =========================
# SETTINGS
# =========================

IMG_SIZE = 224

# Haar cascade for face detection
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)


# =========================
# START WEBCAM
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:
        # Extract face region
        face = frame[y:y+h, x:x+w]

        try:
            # Resize to model input size
            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Normalize same as training
            face_array = face_rgb.astype("float32") / 255.0

            # Add batch dimension
            face_array = np.expand_dims(face_array, axis=0)

            # Predict
            predictions = model.predict(face_array, verbose=0)[0]

            pred_index = int(np.argmax(predictions))
            confidence = float(predictions[pred_index])
            emotion = idx_to_class[pred_index]

            label = f"{emotion}: {confidence*100:.1f}%"

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 255, 0), -1)
            cv2.putText(
                frame,
                label,
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

        except Exception as e:
            print("Prediction error:", e)

    cv2.imshow("Hybrid Emotion Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")