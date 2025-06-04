import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NUM_SAMPLES = 2000

# Rutas
img_dir = '/content/drive/MyDrive/Imagenes'
metadata = pd.read_csv('/content/HAM10000_metadata.csv')

metadata_sample = metadata.sample(n=NUM_SAMPLES, random_state=42).reset_index(drop=True)

IMG_SIZE = 224
def load_images(data, img_dir):
    images = []
    for img_id in data['image_id']:
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
    return np.array(images)

images = load_images(metadata_sample, img_dir)

le = LabelEncoder()
labels = le.fit_transform(metadata_sample['dx'])
labels_cat = to_categorical(labels)

datagen = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8,1.2],
    rescale=1./255
)

X_train, X_val, y_train, y_val = train_test_split(images, labels_cat, test_size=0.2, stratify=labels, random_state=42)

train_gen = datagen.flow(X_train, y_train, batch_size=32, subset='training')
val_gen = datagen.flow(X_val, y_val, batch_size=32, subset='validation', shuffle=False)

# Modelo CNN (DenseNet201 + Transfer Learning)
base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(labels_cat.shape[1], activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15
)

y_pred = []
y_true = []

val_gen.reset()
for i in range(len(val_gen)):
    X_batch, y_batch = val_gen[i]
    preds = model.predict(X_batch)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(y_batch, axis=1))

y_pred = np.array(y_pred)
y_true = np.array(y_true)

unique_labels = np.unique(y_true)
label_names = le.inverse_transform(unique_labels)

history_dict = {
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'model_name': 'DenseNet201'
}

import json
with open(f'/content/drive/MyDrive/model_history_DenseNet201.json', 'w') as f: #Ruta en la que se desea guardar
    json.dump(history_dict, f)

print("Classification Report:")
print(classification_report(y_true, y_pred, labels=unique_labels, target_names=label_names))

print("Confusion Matrix:")
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d",
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
metadata_sample['label_encoded'] = labels
label_dist = metadata_sample['label_encoded'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
sns.barplot(x=le.inverse_transform(label_dist.index), y=label_dist.values)
plt.title("Distribución de clases en la muestra")
plt.xticks(rotation=45)
plt.ylabel("Número de imágenes")
plt.show()