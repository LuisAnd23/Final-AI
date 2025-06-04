import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os
from google.colab import drive
import matplotlib.pyplot as plt


drive.mount('/content/drive')
# Configuración de rutas (AJUSTAR)
DATA_DIR = '/content/drive/MyDrive/'
IMG_DIR = os.path.join(DATA_DIR, 'Imagenes')
METADATA_PATH = os.path.join("/content", 'HAM10000_metadata.csv')

metadata = pd.read_csv(METADATA_PATH)

lesion_types = {
    'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3,
    'akiec': 4, 'vasc': 5, 'df': 6
}
metadata['label'] = metadata['dx'].map(lesion_types)


metadata['age'] = metadata['age'].fillna(metadata['age'].median())
age_scaler = StandardScaler()
metadata['age_norm'] = age_scaler.fit_transform(metadata[['age']])

metadata['sex'] = metadata['sex'].map({'male': 0, 'female': 1}).fillna(0)

location_encoder = OneHotEncoder(sparse_output=False)
location_encoded = location_encoder.fit_transform(metadata[['localization']])
location_cols = [f'loc_{i}' for i in range(location_encoded.shape[1])]
metadata[location_cols] = location_encoded

meta_cols = ['age_norm', 'sex'] + location_cols

def load_image_and_meta(row, img_dir, img_size=(224, 224)):
    img_path = os.path.join(img_dir, row['image_id'] + '.jpg')
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0
    meta = row[meta_cols].values.astype(np.float32)
    label = row['label']
    return img, meta, label

def create_dataset(df, img_dir, batch_size=32):
    def generator():
        for _, row in df.iterrows():
            yield load_image_and_meta(row, img_dir)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(len(meta_cols),), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


train_df, test_df = train_test_split(metadata, test_size=0.2, random_state=42, stratify=metadata['label'])

train_ds = create_dataset(train_df, IMG_DIR)
test_ds = create_dataset(test_df, IMG_DIR)

def build_multimodal_model(num_classes=7):

    img_input = Input(shape=(224, 224, 3), name='image_input')
    base_model = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')(img_input)
    img_features = Dropout(0.3)(base_model)

    img_features = Dense(128, activation='relu')(img_features)

    meta_input = Input(shape=(len(meta_cols),), name='meta_input')
    meta_dense1 = Dense(64, activation='relu')(meta_input)
    meta_dense2 = Dense(128, activation='relu')(meta_dense1)

    combined = Concatenate()([img_features, meta_dense2])

    attention = Dense(256, activation='sigmoid')(combined)
    attended = tf.keras.layers.Multiply()([combined, attention])

    x = Dense(256, activation='relu')(attended)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[img_input, meta_input], outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

model = build_multimodal_model()
model.summary()

def prepare_for_training(ds):
    def process_data(img, meta, label):
        return {'image_input': img, 'meta_input': meta}, label
    return ds.map(process_data)

train_ds_ready = prepare_for_training(train_ds)
test_ds_ready = prepare_for_training(test_ds)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
]

history = model.fit(
    train_ds_ready,
    validation_data=test_ds_ready,
    epochs=15,
    callbacks=callbacks
)


print("\nEvaluación en conjunto de prueba:")
test_loss, test_acc = model.evaluate(test_ds_ready)
print(f"Accuracy: {test_acc*100:.2f}%")


y_true = []
y_pred = []

for (inputs, label) in test_ds_ready.unbatch():
    y_true.append(label.numpy())
    pred = model.predict({'image_input': tf.expand_dims(inputs['image_input'], 0),
                         'meta_input': tf.expand_dims(inputs['meta_input'], 0)}, verbose=0)
    y_pred.append(np.argmax(pred))

history_dict = {
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'model_name': 'Multimodal'
}

import json
with open(f'/content/drive/MyDrive/model_history_Multimodal.json', 'w') as f: # Ruta a la que se desee guardar
    json.dump(history_dict, f)


print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=list(lesion_types.keys())))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()