
# Kuzushiji Image Classifier 

Este proyecto aborda el problema de **clasificación de caracteres japoneses antiguos manuscritos** utilizando el conjunto de datos [Kuzushiji-MNIST](https://github.com/rois-codh/kmnist).  
Forma parte del **Trabajo Práctico N°2** de la materia *Laboratorio de Datos* (2do Cuatrimestre, 2025).

---

## Objetivo

Implementar y evaluar modelos de **aprendizaje supervisado** para tareas de clasificación de imágenes, aplicando técnicas de:

- **Análisis exploratorio de datos (EDA)**  
- **Clasificación binaria y multiclase**  
- **Selección y comparación de modelos**  
- **Validación cruzada (k-fold)**  
- **Evaluación con métricas de desempeño**

---

##  Metodología

### 1. Análisis exploratorio
- Revisión de cantidad de instancias, atributos y clases.
- Visualización de imágenes representativas.
- Análisis de similitudes entre clases.

### 2. Clasificación binaria (clases 4 vs 5)
- Construcción de un subconjunto de datos balanceado.
- Entrenamiento de modelos **KNN** variando cantidad de atributos y vecinos.
- Evaluación con métricas de precisión y recall.

### 3. Clasificación multiclase (10 clases)
- Entrenamiento y comparación de **árboles de decisión** con distintas profundidades.
- Selección del mejor modelo mediante **validación cruzada (k-fold)**.
- Evaluación final sobre un conjunto *held-out*.

---

## Tecnologías utilizadas

- **Python 3.11+**
- **pandas** — manipulación de datos  
- **numpy** — cálculos numéricos  
- **matplotlib / seaborn** — visualización  
- **scikit-learn** — modelos de machine learning y métricas  

