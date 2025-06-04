
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate, Multiply, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import os
from google.colab import drive
import matplotlib.pyplot as plt

def ShuffleNetV2(input_shape=(224, 224, 3), weights='imagenet'):
    """Versión simplificada usando MobileNetV2 como base"""
    return tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )

drive.mount('/content/drive')

# Configuración de rutas (AJUSTAR ESTAS RUTAS)
DATA_DIR = '/content/drive/MyDrive/'
IMG_DIR = os.path.join(DATA_DIR, 'Imagenes')
METADATA_PATH = os.path.join("/content/", 'HAM10000_metadata.csv')

metadata = pd.read_csv(METADATA_PATH)

lesion_types = {
    'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3,
    'akiec': 4, 'vasc': 5, 'df': 6
}
metadata['label'] = metadata['dx'].map(lesion_types)

def load_images(df, img_dir, img_size=(224, 224), sample_fraction=1.0):
    """Carga imágenes con manejo de errores y opción de muestreo"""
    images = []
    labels = []

    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42)

    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row['image_id'] + '.jpg')
        try:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, img_size)
            img = img / 255.0  # Normalización
            images.append(img)
            labels.append(row['label'])
        except Exception as e:
            print(f"Error cargando {img_path}: {str(e)}")
            continue

    return np.array(images), np.array(labels)

train_df, test_df = train_test_split(
    metadata,
    test_size=0.2,
    random_state=42,
    stratify=metadata['label']
)


X_train, y_train = load_images(train_df, IMG_DIR, sample_fraction=0.5)
X_test, y_test = load_images(test_df, IMG_DIR, sample_fraction=0.5)


y_train_onehot = to_categorical(y_train, num_classes=7)
y_test_onehot = to_categorical(y_test, num_classes=7)

def build_combined_model(input_shape=(224, 224, 3), num_classes=7):
    """Arquitectura combinada Xception + ShuffleNet con fusión de características"""

    input_layer = Input(shape=input_shape)

    xception = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )(input_layer)
    xception_pool = GlobalAveragePooling2D()(xception)
    xception_features = Dense(512, activation='relu')(xception_pool)
    shufflenet = ShuffleNetV2(input_shape=input_shape)(input_layer)
    shufflenet_pool = GlobalAveragePooling2D()(shufflenet)
    shufflenet_features = Dense(512, activation='relu')(shufflenet_pool)

    attention = Dense(512, activation='sigmoid')(xception_features)
    attended = Multiply()([xception_features, attention])


    combined = Concatenate()([attended, shufflenet_features])

    x = Dense(256, activation='relu')(combined)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


model = build_combined_model()
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_test, y_test_onehot),
    batch_size=32,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)



print("\nEvaluación en conjunto de prueba:")
test_loss, test_acc = model.evaluate(X_test, y_test_onehot)
print(f"\nAccuracy final: {test_acc*100:.2f}%")


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

history_dict = {
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'model_name': 'XceptionShuffleNet'
}

import json
with open(f'/content/drive/MyDrive/model_history_XceptionShuffleNet.json', 'w') as f: #Rute en la que se desee guarar
    json.dump(history_dict, f)


print("\nReporte de Clasificación Completo:")
print(classification_report(
    y_test,
    y_pred_classes,
    target_names=list(lesion_types.keys())
))

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Exactitud del Modelo')
plt.ylabel('Accuracy')
plt.xlabel('Época')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.ylabel('Loss')
plt.xlabel('Época')
plt.legend()

plt.tight_layout()
plt.show()