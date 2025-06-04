# 🧬 Clasificación de Cáncer de Piel con Redes Neuronales Convolucionales

Este repositorio contiene la implementación de **cuatro modelos de deep learning** para la clasificación automática de lesiones cutáneas utilizando la base de datos **[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)**. Cada modelo emplea una arquitectura distinta, con variantes unimodales y multimodales.
---

## 🧪 Modelos incluidos

| N.º | Modelo | Arquitectura principal | Datos usados | Modalidad |
|-----|--------|-------------------------|--------------|-----------|
| 1️⃣ | `EfficientNetB4` | EfficientNetB4 (imagen) | Imágenes HAM10000 | Unimodal |
| 2️⃣ | `MultimodalEffB0` | EfficientNetB0 + MLP clínico | Imagen + edad, sexo, localización | Multimodal |
| 3️⃣ | `XceptionShuffleNet` | Xception + ShuffleNet (MobileNetV2) | Imagen | Unimodal (fusionado) |
| 4️⃣ | `DenseNet201` | DenseNet201 (con fine-tuning parcial) | Imagen | Unimodal |

---

## 📦 Requisitos

- Python 3.7 o superior
- TensorFlow 2.x
- OpenCV (`cv2`)
- NumPy, Pandas, scikit-learn
- Matplotlib, seaborn
- Google Colab (recomendado por facilidad)

Instala los paquetes con:

```bash
pip install -r requirements.txt


/MyDrive/
│
├── HAM10000_metadata.csv
└── Imagenes/
    ├── ISIC_0024306.jpg
    ├── ...
```

🚀 Cómo ejecutar los modelos en Google Colab

Sigue estos pasos para ejecutar cualquiera de los modelos desde Google Colab:

---

### 1. Montar Google Drive

Primero, monta tu Google Drive para acceder a las imágenes y el archivo CSV.

```python
from google.colab import drive
drive.mount('/content/drive')
 Sube el script del modelo a Google Colab
Abre Google Colab

Haz clic en Archivo > Subir cuaderno

Sube uno de los archivos del repositorio:

modelo_1_efficientnetB4.py

modelo_2_multimodal_efficientnetB0.py

modelo_3_xception_shufflenet.py

modelo_4_densenet201.py
```
4. Ejecuta todas las celdas del script
Asegúrate de que no hay errores en la carga de imágenes ni en la lectura del CSV.

Se entrenará el modelo y se guardará:

El historial (.json)

La matriz de confusión

El reporte de clasificación

Las gráficas de entrenamiento y distribución

5. Revisión de resultados
Al final de la ejecución verás:

Accuracy y loss por época

Reporte de clasificación con precisión, recall y F1-score

Matriz de confusión visual

Distribución de clases usada en la muestra

