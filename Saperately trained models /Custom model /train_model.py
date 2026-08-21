import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# DATASET PATHS
# =========================

train_path = "dataset/train"
test_path = "dataset/test"

# =========================
# IMAGE SETTINGS
# =========================

IMG_SIZE = 48
BATCH_SIZE = 32

# =========================
# DATA PREPROCESSING
# =========================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# =========================
# CNN + ANN MODEL
# =========================

model = Sequential()

# ---------- CNN / DL PART ----------

model.add(Conv2D(
    32,
    (3, 3),
    activation='relu',
    input_shape=(48, 48, 1)
))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(
    64,
    (3, 3),
    activation='relu'
))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(
    128,
    (3, 3),
    activation='relu'
))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# ---------- FLATTEN ----------

model.add(Flatten())

# ---------- ANN PART ----------

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

# OUTPUT LAYER
model.add(Dense(6, activation='softmax'))

# =========================
# MODEL COMPILE
# =========================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# MODEL SUMMARY
# =========================

model.summary()

# =========================
# TRAIN MODEL
# =========================

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=25
)

# =========================
# SAVE MODEL
# =========================

model.save("emotion_model.h5")

print("\nModel Saved Successfully!")

# =========================
# PLOT ACCURACY
# =========================

plt.figure(figsize=(10, 5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(['Train', 'Validation'])

plt.show()

# =========================
# PLOT LOSS
# =========================

plt.figure(figsize=(10, 5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(['Train', 'Validation'])

plt.show()

# =========================
# MODEL EVALUATION
# =========================

predictions = model.predict(test_data)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_data.classes

class_labels = list(test_data.class_indices.keys())

# =========================
# METRICS
# =========================

accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(
    true_classes,
    predicted_classes,
    average='weighted'
)
recall = recall_score(
    true_classes,
    predicted_classes,
    average='weighted'
)
f1 = f1_score(
    true_classes,
    predicted_classes,
    average='weighted'
)

print("\n===== MODEL PERFORMANCE =====")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# =========================
# CLASSIFICATION REPORT
# =========================

print("\n===== CLASSIFICATION REPORT =====\n")

print(classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
))

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()