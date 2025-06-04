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

