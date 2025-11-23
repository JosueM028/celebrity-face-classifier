# celebrity-face-classifier

# Clasificación de Rostros: Portman vs Johansson

Este proyecto realiza un análisis de datos y clasificación binaria de imágenes faciales utilizando **TensorFlow y Python puro**. 

> **Nota:** Por requisitos del proyecto, no se utiliza la librería Scikit-Learn (sklearn) para ninguna etapa del proceso (split, preprocesamiento ni métricas).

## 📂 Dataset
Los datos provienen del [Celebrity Face Image Dataset](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset/data).
Se ha filtrado el dataset para analizar únicamente dos clases:
- Natalie Portman
- Scarlett Johansson

## 🚀 Tecnologías Utilizadas
- **Python 3.x**
- **TensorFlow / Keras**: Para la construcción de la CNN y el pipeline de datos (`tf.data`).
- **OpenCV & NumPy**: Para el análisis estadístico manual y procesamiento de imágenes.
- **Matplotlib**: Para la visualización de datos.

## 📊 Análisis Realizado
1. **Exploración:** Distribución de clases, resoluciones y detección de imágenes corruptas.
2. **Estadística:** Cálculo manual de media y desviación estándar RGB.
3. **Limpieza:** Eliminación de archivos vacíos o formatos no válidos.
4. **Preprocesamiento:** Redimensionamiento, normalización y Data Augmentation sin librerías externas de ML.

