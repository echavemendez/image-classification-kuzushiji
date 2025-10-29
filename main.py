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
#ruta_actual = os.path.dirname(os.path.abspath(__file__))
#carpeta = os.path.join(ruta_actual, "TablasOriginales")
#df_imagenes = pd.read_csv(os.path.join(carpeta, 'kuzushiji_full.csv'))
#df_labels= df_imagenes.iloc[:, [-1]]
#df_imagenes= df_imagenes.drop(df_imagenes.columns[-1], axis=1)
#df_clases = pd.read_csv(os.path.join(carpeta, 'kmnist_classmap_char.csv'))


carpeta = "C:/Users/Reni/Desktop/Labo/tp2/TablasOriginales/"
df_imagenes_labels = pd.read_csv(carpeta + 'kuzushiji_full.csv')
df_clases = (carpeta + 'kmnist_classmap_char.csv')
df_labels= df_imagenes.iloc[:, [-1]]
df_imagenes = df_imagenes.drop(df_imagenes.columns[-1], axis=1)
df_clases = pd.read_csv(carpeta + 'kmnist_classmap_char.csv')

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

#%% 
# ---  VISUALIZACIÓN DE ALGUNAS IMÁGENES ---
# Para hacernos una idea visual de cómo son los caracteres,
# vamos a tomar un representante de cada clase para ver a grandes rasgos como son.
# Creamos el mapeo de clase → carácter japonés
mapa_clases = dict(zip(df_clases['class'], df_clases['char']))

# Índices de las filas que queremos mostrar
indices = [2, 3, 5, 98, 225, 329, 320, 411, 406, 407]

# Clases correspondientes a cada imagen
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Ojo: corregí el 411 que dijiste “6” pero debería ser “7”

# Creamos la figura y los subplots (2 filas x 5 columnas)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

# Recorremos las imágenes y las mostramos haciendo uso de la funcion dada en el enunciado del tp con un forloop
for i, ax in enumerate(axes.flat):
    idx = indices[i]
    img = np.array(df_imagenes.iloc[idx]).reshape((28, 28))
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Clase {labels[i]} ({mapa_clases[labels[i]]})", fontsize=10)
    ax.axis('off')  # Sacamos los ejes para que quede más limpio

# Ajustamos espacios para que no se solapen los títulos
plt.suptitle("Ejemplos representativos de cada clase en el dataframe", fontsize=14, y=1.03)
plt.tight_layout()
plt.show()

#%%
# También compararemos imagenes con misma etiqueta para entender que tanto difieren, elegimos una al azar

# Índices de las imágenes que queremos mostrar (todas de la clase 4)
indices = [4, 6, 77, 154]
clase = 4

fig, axes = plt.subplots(1, 4, figsize=(8, 3))

# Recorremos y mostramos cada imagen
for i, ax in enumerate(axes.flat):
    idx = indices[i]
    img = np.array(df_imagenes.iloc[idx]).reshape((28, 28))
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Clase {clase}", fontsize=10)
    ax.axis('off')

# Ajustamos espacios y mostramos
plt.suptitle(f"Comparación de imágenes de la clase {clase}", fontsize=14, y=1.05)
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

######


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

######no nos sirve toma pixeles aleatorios, entonces cada vez que lo ejecutamos nos da un gráfico distintoo y no podemos concluir nada sobre el 

#%% 
# --- CORRELACIÓN ENTRE VARIABLES --*
# En este gráfico mostramos la correlación entre los pixeles

# Calcula la intensidad promedio de CADA píxel (columna) a lo largo de TODAS las imágenes (filas),
# convierte a array de NumPy y lo reacomoda a una matriz 28x28 para visualizar como imagen.
mean_image = df_imagenes.mean(axis=0).values.reshape(28, 28)

# Calcula el desvío estándar (variabilidad) de CADA píxel entre imágenes,
# lo pasa a array y lo reacomoda a 28x28 para graficar.
std_image = df_imagenes.std(axis=0).values.reshape(28, 28)

# Muestra la imagen promedio en escala de grises (zonas claras = píxeles que suelen encenderse).
plt.imshow(mean_image, cmap="gray")
# Agrega título al gráfico actual.
plt.title("Imagen promedio")
# Renderiza el gráfico en pantalla.
plt.show()


# Para entender el gráfico! 
# Es una imagen de 28x28, entonces, re distribuye los valores del 0 al 783 de la siguiente forma:
# 0   1   2   3   ...  27
# 28 29  30  31  ...  55
# 56 57  58  59  ...  83
# ...
# 756 ................. 783

# Muestra el mapa de desviación estándar (color más claro = más variación entre imágenes).
plt.imshow(std_image, cmap="hot")
# Agrega título al gráfico actual.
plt.title("Desviación estándar por píxel")
# Renderiza el gráfico en pantalla.
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
#%%
#==================== Comenzamos con el ejercicio 2 =======================================
# 2. a)
# Armamos un dataframe solamente con las filas de clases 4 y 5 

query_cuatro_y_cinco = """
SELECT *
FROM df_imagenes_labels il
WHERE (label = 4 
       OR
       label = 5)
"""

df_cuatro_y_cinco = dd.query(query_cuatro_y_cinco).df()
    
print(df_cuatro_y_cinco)

# Muestra 14000 filas, entonces, tenemos 14000 muestras.

#==========================================

# Me fijo si está balanceado o no. 

query_balanceado = """
SELECT
  SUM(CASE WHEN label = '4' THEN 1 ELSE 0 END) AS 'Cantidad de Clase 4',
  SUM(CASE WHEN label = '5' THEN 1 ELSE 0 END) AS 'Cantidad de Clase 5'
FROM df_cuatro_y_cinco;
"""

df_balanceado = dd.query(query_balanceado).df()

# Tengo 7000 de clase 4 y 7000 de clase 5, entonces está balanceado.

#%%
