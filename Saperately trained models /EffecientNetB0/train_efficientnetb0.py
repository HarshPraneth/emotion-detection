import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(BASE_DIR, "dataset", "train")
test_path = os.path.join(BASE_DIR, "dataset", "test")
models_path = os.path.join(BASE_DIR, "models")

os.makedirs(models_path, exist_ok=True)

model_save_path = os.path.join(models_path, "efficientnetb0_emotion_model.keras")
labels_save_path = os.path.join(models_path, "efficientnetb0_class_labels.json")

accuracy_graph_path = os.path.join(models_path, "efficientnetb0_accuracy.png")
loss_graph_path = os.path.join(models_path, "efficientnetb0_loss.png")
confusion_matrix_path = os.path.join(models_path, "efficientnetb0_confusion_matrix.png")

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10

# =========================
# Check paths
# =========================
print("Train path:", train_path)
print("Test path:", test_path)
print("Models path:", models_path)

print("Train path exists:", os.path.exists(train_path))
print("Test path exists:", os.path.exists(test_path))
print("Models path exists:", os.path.exists(models_path))

if not os.path.exists(train_path):
    raise FileNotFoundError("Train path not found. Check dataset/train folder.")

if not os.path.exists(test_path):
    raise FileNotFoundError("Test path not found. Check dataset/test folder.")

# =========================
# Data generators
# EfficientNetB0 uses preprocess_input
# =========================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = len(train_data.class_indices)

print("\nClass labels:", train_data.class_indices)
print("Number of classes:", NUM_CLASSES)

# =========================
# Save class labels
# =========================
with open(labels_save_path, "w") as f:
    json.dump(train_data.class_indices, f)

print("Class labels saved at:", labels_save_path)

# =========================
# Image count per class
# =========================
print("\n===== TRAIN IMAGE COUNT =====")
for cls, idx in train_data.class_indices.items():
    count = np.sum(train_data.classes == idx)
    print(cls, ":", count)

print("\n===== TEST IMAGE COUNT =====")
for cls, idx in test_data.class_indices.items():
    count = np.sum(test_data.classes == idx)
    print(cls, ":", count)

# =========================
# Class weights
# Helps if model predicts only one class
# =========================
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights_array))
print("\nClass weights:", class_weights)

# =========================
# EfficientNetB0 base model
# =========================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# =========================
# Custom classifier
# =========================
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# Callbacks
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    model_save_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# =========================
# Train model
# =========================
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weights
)

print("\nEfficientNetB0 training completed.")
print("Best model saved at:", model_save_path)

# =========================
# Accuracy graph
# =========================
plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("EfficientNetB0 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.grid(True)
plt.savefig(accuracy_graph_path)
plt.show()

print("Accuracy graph saved at:", accuracy_graph_path)

# =========================
# Loss graph
# =========================
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("EfficientNetB0 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.grid(True)
plt.savefig(loss_graph_path)
plt.show()

print("Loss graph saved at:", loss_graph_path)

# =========================
# Evaluation
# =========================
test_data.reset()

predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(
    true_classes,
    predicted_classes,
    average="weighted",
    zero_division=0
)
recall = recall_score(
    true_classes,
    predicted_classes,
    average="weighted",
    zero_division=0
)
f1 = f1_score(
    true_classes,
    predicted_classes,
    average="weighted",
    zero_division=0
)

print("\n===== EfficientNetB0 PERFORMANCE =====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\n===== EfficientNetB0 CLASSIFICATION REPORT =====\n")
print(classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels,
    zero_division=0
))

# =========================
# Confusion matrix
# =========================
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("EfficientNetB0 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(confusion_matrix_path)
plt.show()

print("Confusion matrix saved at:", confusion_matrix_path)

print("\nAll files saved inside models folder.")