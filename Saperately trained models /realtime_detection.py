import os
import cv2
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# =========================
# FIX OLD H5 BATCH NORMALIZATION ISSUE
# =========================

class FixedBatchNormalization(BatchNormalization):
    def __init__(self, *args, **kwargs):
        kwargs.pop("renorm", None)
        kwargs.pop("renorm_clipping", None)
        kwargs.pop("renorm_momentum", None)
        super().__init__(*args, **kwargs)

# =========================
# BASE PATH
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# =========================
# MODEL PATHS
# =========================

MODEL_PATHS = {
    "1": {
        "name": "Custom CNN + ANN",
        "path": os.path.join(MODEL_DIR, "emotion_model.h5"),
        "labels_path": None,
        "img_size": 48,
        "color_mode": "grayscale",
        "model_type": "old_h5",
        "preprocess": "rescale"
    },
    "2": {
        "name": "VGG16",
        "path": os.path.join(MODEL_DIR, "vgg16_emotion_model2.keras"),
        "labels_path": os.path.join(MODEL_DIR, "vgg16_class_labels.json"),
        "img_size": 224,
        "color_mode": "rgb",
        "model_type": "keras",
        "preprocess": "rescale"
    },
    "3": {
        "name": "MobileNetV2",
        "path": os.path.join(MODEL_DIR, "mobilenetv2_emotion_model.keras"),
        "labels_path": os.path.join(MODEL_DIR, "mobilenetv2_class_labels.json"),
        "img_size": 224,
        "color_mode": "rgb",
        "model_type": "keras",
        "preprocess": "mobilenet"
    },
    "4": {
        "name": "ResNet50",
        "path": os.path.join(MODEL_DIR, "resnet50_emotion_model.keras"),
        "labels_path": os.path.join(MODEL_DIR, "resnet50_class_labels.json"),
        "img_size": 224,
        "color_mode": "rgb",
        "model_type": "keras",
        "preprocess": "resnet"
    },
    "5": {
        "name": "EfficientNetB0",
        "path": os.path.join(MODEL_DIR, "efficientnetb0_emotion_model.keras"),
        "labels_path": os.path.join(MODEL_DIR, "efficientnetb0_class_labels.json"),
        "img_size": 224,
        "color_mode": "rgb",
        "model_type": "keras",
        "preprocess": "efficientnet"
    }
}

# =========================
# DEFAULT EMOTION LABELS
# Used only if labels JSON is missing
# =========================

default_emotions = [
    "Angry",
    "Disgust",
    "Happy",
    "Neutral",
    "Sad",
    "Shock"
]

# =========================
# SELECT MODEL
# =========================

print("\nSelect Model for Real-Time Emotion Detection")
print("1. Custom CNN + ANN")
print("2. VGG16")
print("3. MobileNetV2")
print("4. ResNet50")
print("5. EfficientNetB0")

choice = input("\nEnter model number: ")

if choice not in MODEL_PATHS:
    print("Invalid choice")
    exit()

selected_model = MODEL_PATHS[choice]

print(f"\nLoading model: {selected_model['name']}")
print("Model path:", selected_model["path"])

if not os.path.exists(selected_model["path"]):
    print("\nModel file not found!")
    print("Check exact file name inside models folder:")
    print(MODEL_DIR)
    exit()

# =========================
# LOAD MODEL
# =========================

try:
    if selected_model["model_type"] == "old_h5":
        model = load_model(
            selected_model["path"],
            custom_objects={
                "BatchNormalization": FixedBatchNormalization
            },
            compile=False
        )
    else:
        model = load_model(selected_model["path"], compile=False)

except Exception as e:
    print("\nError while loading model:")
    print(e)
    exit()

IMG_SIZE = selected_model["img_size"]
COLOR_MODE = selected_model["color_mode"]
PREPROCESS_TYPE = selected_model["preprocess"]

# =========================
# LOAD CLASS LABELS
# =========================

labels_path = selected_model["labels_path"]

if labels_path is not None and os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        class_indices = json.load(f)

    emotions = [None] * len(class_indices)

    for class_name, index in class_indices.items():
        emotions[index] = class_name.capitalize()

    print("\nLoaded class labels from:", labels_path)

else:
    emotions = default_emotions
    print("\nUsing default class labels.")

print("Class labels:", emotions)

print("\nModel loaded successfully!")
print("Input size:", IMG_SIZE)
print("Color mode:", COLOR_MODE)
print("Preprocessing:", PREPROCESS_TYPE)
print("Press 'q' to quit")

# =========================
# FACE DETECTOR
# =========================

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_classifier.empty():
    print("Error: Face detector not loaded")
    exit()

# =========================
# START WEBCAM
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not detected")
    print("On Mac, allow Camera permission for Terminal or VS Code.")
    exit()

# =========================
# PREPROCESS FUNCTION
# =========================

def preprocess_face(face_roi, gray_roi, img_size, color_mode, preprocess_type):

    # =========================
    # Custom CNN
    # 48x48 grayscale
    # =========================
    if color_mode == "grayscale":
        roi = cv2.resize(gray_roi, (img_size, img_size))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        return roi

    # =========================
    # RGB models
    # =========================
    roi = cv2.resize(face_roi, (img_size, img_size))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi.astype("float32")

    if preprocess_type == "rescale":
        roi = roi / 255.0

    elif preprocess_type == "mobilenet":
        roi = mobilenet_preprocess(roi)

    elif preprocess_type == "resnet":
        roi = resnet_preprocess(roi)

    elif preprocess_type == "efficientnet":
        roi = efficientnet_preprocess(roi)

    else:
        roi = roi / 255.0

    roi = np.expand_dims(roi, axis=0)
    return roi

# =========================
# REAL-TIME LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        face_roi = frame[y:y + h, x:x + w]
        gray_roi = gray[y:y + h, x:x + w]

        roi = preprocess_face(
            face_roi=face_roi,
            gray_roi=gray_roi,
            img_size=IMG_SIZE,
            color_mode=COLOR_MODE,
            preprocess_type=PREPROCESS_TYPE
        )

        # =========================
        # PREDICTION
        # =========================

        prediction = model.predict(roi, verbose=0)

        max_index = np.argmax(prediction[0])
        emotion_label = emotions[max_index]
        confidence = prediction[0][max_index] * 100

        text = f"{selected_model['name']}: {emotion_label} ({confidence:.1f}%)"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Multi-Model Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# RELEASE
# =========================

cap.release()
cv2.destroyAllWindows()