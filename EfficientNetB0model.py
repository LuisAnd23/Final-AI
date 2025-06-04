import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

tf.config.optimizer.set_jit(True)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from google.colab import drive
drive.mount('/content/drive')


DATA_DIR = '/content/drive/MyDrive/'  # ruta
IMG_DIR = os.path.join(DATA_DIR, 'Imagenes') # Ruta en la que se encuentran todas las imagenes
METADATA_PATH = os.path.join("/content/", 'HAM10000_metadata.csv') # Ruta en la que se encuentra la metadata con las etiquetas

metadata = pd.read_csv(METADATA_PATH)
lesion_types = {
    'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3,
    'akiec': 4, 'vasc': 5, 'df': 6
}
metadata['label'] = metadata['dx'].map(lesion_types)

sample_df = metadata.sample(frac=0.2, random_state=42)

def fast_image_loader(df, img_dir, batch_size=32, img_size=(192, 192)):
    while True:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            images = []
            labels = []
            for _, row in batch.iterrows():
                img_path = os.path.join(img_dir, row['image_id'] + '.jpg')
                img = tf.io.read_file(img_path)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, img_size)
                img = img / 255.0
                images.append(img)
                labels.append(row['label'])
            yield tf.stack(images), tf.keras.utils.to_categorical(labels, num_classes=7)


train_df, test_df = train_test_split(sample_df, test_size=0.2, random_state=42, stratify=sample_df['label'])

def build_light_model():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(192, 192, 3),
        pooling='avg'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_light_model()
model.summary()

BATCH_SIZE = 16
train_steps = len(train_df) // BATCH_SIZE
val_steps = len(test_df) // BATCH_SIZE

history = model.fit(
    fast_image_loader(train_df, IMG_DIR, BATCH_SIZE),
    steps_per_epoch=train_steps,
    epochs=15,
    validation_data=fast_image_loader(test_df, IMG_DIR, BATCH_SIZE),
    validation_steps=val_steps,
    verbose=1
)

test_gen = fast_image_loader(test_df, IMG_DIR, BATCH_SIZE)
y_true = []
y_pred = []

for _ in range(val_steps):
    x, y = next(test_gen)
    y_true.extend(np.argmax(y, axis=1))
    y_pred.extend(np.argmax(model.predict(x), axis=1))

history_dict = {
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'model_name': 'EfficientNetB0'
}

import json
with open(f'/content/drive/MyDrive/model_history_EfficientNetB0.json', 'w') as f: #cambiar a ruta donde se desee guardar
    json.dump(history_dict, f)


print("\nReporte Final:")
print(classification_report(y_true, y_pred, target_names=list(lesion_types.keys())))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()
plt.show()