import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DATASET_PATH = "dataset"
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 6
# Standardized categories to match app.py
CATEGORIES = ["Cataract", "Diabetic_Retinopathy", "Glaucoma", "Normal"]
MODEL_SAVE_PATH = "models/eye_disease_model.h5"

# Data Preprocessing
logging.info("Initializing data generators...")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="training",
    classes=CATEGORIES
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="validation",
    classes=CATEGORIES,
    shuffle=False # No shuffle for reliable evaluation
)

# Load Pretrained Model
logging.info("Loading Xception model...")
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(CATEGORIES), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- FIX: Define checkpoint BEFORE training ---
os.makedirs("models", exist_ok=True)
checkpoint = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
    mode='max' # Save on max validation accuracy
)

# --- FIX: Train the model only ONCE ---
logging.info(f"Starting model training for {EPOCHS} epochs...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# Load the best model saved by the checkpoint for evaluation
logging.info("Loading best saved model for final evaluation...")
best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# Evaluate Model
logging.info("Evaluating model on validation data...")
val_preds = best_model.predict(val_data)
y_true = val_data.classes
y_pred = np.argmax(val_preds, axis=1)

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=CATEGORIES))
logging.info("Model evaluation complete.")

# Accuracy and Loss Plots
plt.figure(figsize=(14, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)

plt.show()