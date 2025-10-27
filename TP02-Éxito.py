# Grupo: Éxito
# Integrantes:
# - Barello, Renata — LU: 1060/24
# - Echave Méndez, Manuel — LU: 1333/23
# - González Frey, Paloma — LU: 1030/24
#
# Trabajo Práctico N°2 — Laboratorio de Datos
# 2do Cuatrimestre 2025
#
# Descripción:
# En este trabajo práctico se aborda el problema de clasificación de imágenes
# utilizando el conjunto de datos Kuzushiji-MNIST, compuesto por caracteres
# japoneses antiguos manuscritos. El objetivo principal es aplicar técnicas de
# aprendizaje supervisado, específicamente métodos de clasificación, evaluación
# y selección de modelos, mediante validación cruzada.
#
# A lo largo del código se desarrolla:
#  - Un análisis exploratorio de los datos.
#  - Una clasificación binaria entre las clases 4 y 5 empleando K-Nearest Neighbors.
#  - Una clasificación multiclase con árboles de decisión y validación cruzada.
#
# Se busca evaluar el desempeño de distintos modelos y configuraciones,
# justificando las decisiones en base a métricas y visualizaciones obtenidas.
#%%
# imports necesarios.

import duckdb as dd
import pandas as pd
import os
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#%%
# paths
ruta_actual = os.path.dirname(os.path.abspath(__file__))
carpeta = os.path.join(ruta_actual, "TablasOriginales")
df_imagenes = pd.read_csv(os.path.join(carpeta, 'kuzushiji_full.csv'))
df_clases = pd.read_csv(os.path.join(carpeta, 'kmnist_classmap_char.csv'))

#%%
#%% 
# ======================================================
# ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ======================================================
# En esta sección analizamos la estructura general del dataset,
# su distribución, balance de clases y algunas visualizaciones básicas
# que nos ayuden a entender mejor la información antes de modelar.
# ======================================================

# ---  VISTA GENERAL ---
# Empezamos mirando la estructura básica para tener una idea del contenido.
print("Dimensiones del dataset de imágenes:", df_imagenes.shape)
print("Cantidad de etiquetas:", df_clases['class'].shape)
print("\nPrimeras filas del dataset:")
print(df_imagenes.head())

#%% 
# ---  INFORMACIÓN GENERAL ---
# Vemos los tipos de datos y si hay valores faltantes
print("\nInformación general de df_imagenes:")
df_imagenes.info()
print("\nValores faltantes por columna:")
print(df_imagenes.isnull().sum().sum())  # Total de valores faltantes en el dataset

#%% 
# ---  ANÁLISIS DE LA VARIABLE OBJETIVO ---
# Analizamos cómo están distribuidas las clases (si el dataset está balanceado o no)
print("\nDistribución de clases:")
print(df_clases['class'].value_counts())

img = np.array(df_imagenes.iloc[12]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()

#%% 
# ---  VISUALIZACIÓN DE ALGUNAS IMÁGENES ---
# Para hacernos una idea visual de cómo son los caracteres,
# mostramos algunas imágenes aleatorias del dataset.

indices = np.random.choice(len(df_imagenes), size=9, replace=False)
plt.figure(figsize=(6,6))
for i, idx in enumerate(indices):
    img = np.array(df_imagenes.iloc[idx]).reshape(28, 28)
    plt.subplot(3,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Clase {df_clases.iloc[idx,0]}")
plt.suptitle("Ejemplos de imágenes del dataset", fontsize=14)
plt.tight_layout()
plt.show()

#%% 
# ---  ANÁLISIS ESTADÍSTICO BÁSICO ---
# Calculamos algunas métricas descriptivas para ver la dispersión y rango
# de valores en los píxeles. Esto nos puede servir para detectar columnas inútiles
# (por ejemplo, píxeles que siempre son 0 o tienen poca variación).

print("\nResumen estadístico de los valores de píxeles:")
print(df_imagenes.describe())

# Vemos cuántos píxeles son completamente negros (0) en todas las imágenes
pixeles_constantes = (df_imagenes.var() == 0).sum()
print(f"\nCantidad de píxeles sin variación (constantes): {pixeles_constantes}")

#%% 
# ---  CORRELACIÓN ENTRE VARIABLES (opcional) ---
# En datasets grandes de imágenes esto puede ser pesado, pero podemos
# probar con un subconjunto para explorar correlaciones entre píxeles.

sample = df_imagenes.sample(500)  # tomamos una muestra chica para hacerlo manejable
corr = sample.corr().iloc[:10, :10]  # solo mostramos una parte
plt.figure(figsize=(6,5))
sns.heatmap(corr, cmap="coolwarm", square=True)
plt.title("Ejemplo de correlación entre algunos píxeles")
plt.show()

#%% 
# ---  CONCLUSIONES PRELIMINARES ---
# A partir de este análisis podremos sacar algunas conclusiones iniciales, por ejemplo:
# - Si el dataset está balanceado o hay clases sobrerrepresentadas.
# - Si hay variables (píxeles) con muy poca información útil.
# - Qué tan distinguibles parecen las clases a simple vista.
# Estas observaciones van a servirnos para decidir cómo reducir atributos
# o qué modelos de clasificación probar más adelante.

#%%

MSE_k = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [10, 0], [15, 0], [20, 0], [25, 0]]


for tupla in MSE_k:
    knn = KNeighborsRegressor(n_neighbors=tupla[0])
    
    knn.fit(df_filtrados[['altura_madre']], df_filtrados['altura_hijo'])
    
    ypred = knn.predict(df_filtrados[['altura_madre']])
    
    tupla[1]=mean_squared_error(df_filtrados['altura_hijo'], ypred)
    
    np.sqrt(mean_squared_error(df_filtrados['altura_hijo'], ypred))

    
