# 🎭 Sistema de Clasificación Facial de Celebridades

Este repositorio contiene un proyecto integral de visión por computadora desarrollado con **TensorFlow y Python puro**. El sistema es capaz de realizar reconocimiento facial tanto binario como multiclase, implementando arquitecturas de redes neuronales convolucionales (CNN) desde cero.

> ⚠️ **Restricción Técnica:** Por requisitos estrictos del proyecto, **NO se utiliza Scikit-Learn (sklearn)** en ninguna etapa. Toda la división de datos, preprocesamiento, aumento de datos y cálculo de métricas (incluyendo matrices de confusión) se implementó manualmente o utilizando utilidades nativas de TensorFlow/Keras.

## 🚀 Características del Proyecto

El proyecto se divide en tres módulos principales:

### 1. Clasificación Binaria (Custom CNN)
* **Objetivo:** Distinguir entre Natalie Portman y Scarlett Johansson.
* **Modelo:** Arquitectura CNN personalizada ligera.
* **Técnicas:** Limpieza de datos corruptos con OpenCV, Data Augmentation, y manejo de desbalance de clases mediante `class_weights`.

### 2. Clasificación Multiclase (VGG16 desde cero)
* **Objetivo:** Clasificar rostros de 6 celebridades:
    * Denzel Washington
    * Hugh Jackman
    * Jennifer Lawrence
    * Megan Fox
    * Natalie Portman
    * Scarlett Johansson
* **Modelo:** Implementación manual de la arquitectura **VGG16** (sin Transfer Learning), optimizada con **Batch Normalization** para evitar el colapso de modo.
* **Evaluación:** Matriz de confusión multiclase generada con `tf.math.confusion_matrix`.

### 3. Aplicación Web (Deployment)
* Interfaz web construida con **Flask**.
* Permite a los usuarios subir una imagen y obtener una predicción en tiempo real utilizando el modelo entrenado.

