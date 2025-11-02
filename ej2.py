
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

#%%
# paths
#ruta_actual = os.path.dirname(os.path.abspath(__file__))
#carpeta = os.path.join(ruta_actual, "TablasOriginales")
#df_imagenes_labels = pd.read_csv(os.path.join(carpeta, 'kuzushiji_full.csv'))
#df_labels= df_imagenes_labels.iloc[:, [-1]]
#df_imagenes= df_imagenes_labels.drop(df_imagenes_labels.columns[-1], axis=1)
#df_clases = pd.read_csv(os.path.join(carpeta, 'kmnist_classmap_char.csv'))


carpeta = ("C:/Users/Reni/Desktop/Labo/tp2/TablasOriginales/")
df_imagenes_labels = pd.read_csv(carpeta + 'kuzushiji_full.csv')
df_labels= df_imagenes_labels.iloc[:, [-1]]
df_imagenes= df_imagenes_labels.drop(df_imagenes_labels.columns[-1], axis=1)
df_clases = pd.read_csv(carpeta + 'kmnist_classmap_char.csv')

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
# Notamos que todos los atributos tienen valores no nulos
#%%
# ---  ANÁLISIS DE LA VARIABLE OBJETIVO ---
# Contamos cuántas muestras hay de cada etiqueta (0 a 9)


cantidad_de_muestras = dd.sql("""
                SELECT label, COUNT(*) AS apariciones
                FROM df_labels
                GROUP BY label
                ORDER BY label
                """).df()

promedio = cantidad_de_muestras['apariciones'].sum() / 10

#%% Graficamos los resultados

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x="label", y="apariciones", data=cantidad_de_muestras,ax=ax)

# Línea del promedio
ax.axhline(promedio, color='red', linewidth=2, linestyle='--', label='Promedio')

# Etiquetas y formato
ax.set_xlabel("Etiqueta (label)", fontsize=12)
ax.set_ylabel("Cantidad de apariciones", fontsize=12)
ax.set_title("Distribución de la variable objetivo", fontsize=14)
ax.legend()

plt.tight_layout()
plt.show()

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
# También compararemos imagenes con misma etiqueta para entender que tanto difieren, elegimos una al azar (ejemplo: clase 8). [1.c)]

# Filtramos las filas que corresponden a la clase 8
df_clase8 = dd.sql("""
    SELECT *
    FROM df_imagenes_labels
    WHERE label = 8
""").df()

# Tomamos algunos índices aleatorios de esa clase
muestras = df_clase8.sample(40, random_state=42)  # elegimos 40 imágenes al azar
muestras_sin_label = muestras.iloc[:, :-1].copy() # para poder visualizar bien
clase = 8

# Graficamos las imágenes seleccionadas
fig, axes = plt.subplots(5, 8, figsize=(10, 6))  # organizamos en 5 filas y 8 columnas

for i, ax in enumerate(axes.flat):
    img = np.array(muestras_sin_label.iloc[i]).reshape((28, 28))
    ax.imshow(img, cmap='gray')
    ax.set_title(f"", fontsize=8)
    ax.axis('off')

plt.suptitle(f"Comparación de imágenes de la clase {clase}", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

#%% Ordenamos el dataset

df_imagenes_labels_ord = dd.sql("""
                                SELECT *
                                FROM df_imagenes_labels 
                                ORDER BY label
                                """).df()
                
#%% Calculamos la imagen promedio para cada clase

m_prom_clase = [np.zeros((28,28)) for _ in range(10)] # Creamos 10 matrices inicializadas en 0

for etiqueta in range(10):  # Iteramos del 0 al 9
    subset = df_imagenes_labels_ord[df_imagenes_labels_ord['label'] == etiqueta]  # Filtramos por cada label
    for i in range(len(subset)):
        matriz = np.array(subset.iloc[i, 0:784]).reshape(28, 28)
        m_prom_clase[etiqueta] += matriz
    m_prom_clase[etiqueta] /= len(subset)
    
#%% Graficamos los resultados

fig, axes = plt.subplots(1,10, figsize =(20,5))
for i in range (10):
    axes[i].imshow(m_prom_clase[i], cmap='gray')
    axes[i].axis('off')
plt.show()
#%%
# Queremos ver si es mas facil diferenciar entre una imagen de la clase 2 y la 1, o de la 2 y la 6.
# Para esto vamos a ver cual de las diferencias entre las imagenes promedio de las clases dadas se acerca mas a la matriz nula.

matriz_1= m_prom_clase[1]
matriz_2= m_prom_clase[2]
matriz_6=m_prom_clase[6]

diferencia_1_2= matriz_1 - matriz_2
diferencia_6_2= matriz_6 - matriz_2

# Vamos a utilizar la norma Frobenius para medir distancias entre matrices y obtener un escalar real para comparar mas facilmente

dist_1_2 = np.linalg.norm(diferencia_1_2, 'fro')
dist_6_2 = np.linalg.norm(diferencia_6_2, 'fro')

# ¿Cuales son mas parecidas?

if dist_1_2 > dist_6_2:
    print("Son mas parecidas las clases 2 y 6")
else:
    print("Son mas parecidas las clases 2 y 1")
    
# Concluimos que son mas similares las clases 2 y 1.

#%%
# Graficamos las imagenes de las restas
fig, ax = plt.subplots()
ax.imshow(diferencia_1_2,cmap='gray')
plt.suptitle(f"Similitud promedio de imagenes clase 1 - clase2", fontsize=14, y=1.02)
ax.axis('off')

fig, ax = plt.subplots()
ax.imshow(diferencia_6_2,cmap='gray')
plt.suptitle(f"Similitud promedio de imagenes clase 6 - clase2", fontsize=14, y=1.02)
ax.axis('off')

#%% Calculamos la imagen promedio total

matriz_prom = np.zeros((28,28))
for i in range (10):
    matriz_prom += m_prom_clase[i]
matriz_prom /= 10
    
#%% Graficamos los resultados

fig, ax = plt.subplots()
ax.imshow(matriz_prom, cmap='gray')
ax.axis('off')

#%%
# Calculamos imagen promedio de la clase 8 especificamente, para referenciarla a posteriori en el informe.

matriz_8= m_prom_clase[8]
fig, ax = plt.subplots()
ax.imshow(matriz_8,cmap='gray')
plt.suptitle(f"Imagen promedio de la clase 8", fontsize=14, y=1.02)
ax.axis('off')
#%% 
# --- VARIABILIDAD ENTRE PIXELES IMAGEN-IMAGEN --*
# En este gráfico mostramos la variabilidad entre los pixeles de todas las imagenes

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

# Muestra el mapa de desviación estándar (color más claro = más variación entre imágenes, blanco siendo mas variable, negro siendo menos variable, pasando por rojo y amarillo).
plt.imshow(std_image, cmap="hot")
# Agrega título al gráfico actual.
plt.title("Desviación estándar por píxel")
# Renderiza el gráfico en pantalla.
plt.show()
#%%
# Decidimos descartar 4 pixeles en cada margen de las imagenes, sacando conclusiones gracias a las previas visualizaciones realizadas.

# Vamos a tener el df con imagenes de 27 columnas y 20 filas una vez elimine las 4 filas superiores y las 4 inferiores
# Nos armamos una lista con los indices que vamos a necesitar borrar para descartar los datos,
# Son los 4 primeros y los 4 ultimos de cada fila
# Cada fila tiene 28 de largo, voy a descartar el [0-3] y el [24-27], y las primeras y ultimas 4 filas.

columnas_a_descartar = []

for fila in range(28):  # recorremos cada fila original
    if fila < 4 or fila >= 24:
        # filas que eliminamos enteras (las 4 de arriba y 4 de abajo)
        columnas_a_descartar.extend(range(fila * 28, (fila + 1) * 28))
    else:
        # en las filas del medio eliminamos 4 primeras y 4 últimas columnas
        inicio = fila * 28
        columnas_a_descartar.extend(range(inicio, inicio + 4))      # 4 primeras
        columnas_a_descartar.extend(range(inicio + 24, inicio + 28))  # 4 últimas

# eliminamos esas columnas del DataFrame original
df_datos_calidad = df_imagenes_labels.drop(df_imagenes_labels.columns[columnas_a_descartar], axis=1)
#%%
# Visualizacion de nuestro nuevo df
matriz_filtrada2 = matriz_prom[4:-4, 4:-4]
fig, ax = plt.subplots()
ax.imshow(matriz_filtrada2, cmap='gray')
ax.axis('off')
#%%
#==================== Comenzamos con el ejercicio 2 =======================================
# 2. a)
# Armamos un dataframe solamente con las filas de clases 4 y 5 

query_cuatro_y_cinco = """
SELECT *
FROM df_datos_calidad 
WHERE (label = 4 
       OR
       label = 5)
"""

df_cuatro_y_cinco = dd.query(query_cuatro_y_cinco).df()
    
print(df_cuatro_y_cinco)

# Sabemos que esta balanceado porque previamente lo calculamos.

#%%
#2b) Separamos datos de prueba y datos de entrenamiento
# Separamos las features (X) y las etiquetas (y)
X = df_cuatro_y_cinco.drop(columns='label')
y = df_cuatro_y_cinco['label']

# 80% para entrenamiento y 20% para test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # proporción de test
    random_state=42,     # para reproducibilidad
    stratify=y           # mantiene el balance entre clases 4 y 5
)

print("Tamaño train:", X_train.shape[0])
print("Tamaño test:", X_test.shape[0])
print("\nDistribución de clases en train:")
print(y_train.value_counts(normalize=True))
print("\nDistribución de clases en test:")
print(y_test.value_counts(normalize=True))

#%%
#2c)

#Elegimos los atributos para llevar a cabo nuestro modelo knn y comparar 

# --- VARIABILIDAD ENTRE PIXELES IMAGEN-IMAGEN --*
# En este gráfico mostramos la variabilidad entre los pixeles de todas las clases 4 y 5.
# Repito el grafico de desviacion por pixel para elegir conjuntos de 3 mas relevantes. 
df_sin_ultima = df_cuatro_y_cinco.drop(df_cuatro_y_cinco.columns[-1], axis=1)
desviacion_estandard = df_sin_ultima.std(axis=0).values.reshape(20, 20)

# Calcula la intensidad promedio de CADA píxel (columna) a lo largo de TODAS las imágenes (filas),
# convierte a array de NumPy y lo reacomoda a una matriz 28x28 para visualizar como imagen.
mean_image = df_sin_ultima.mean(axis=0).values.reshape(20, 20)

# Calcula el desvío estándar (variabilidad) de CADA píxel entre imágenes,
# lo pasa a array y lo reacomoda a 28x28 para graficar.
std_image = df_sin_ultima.std(axis=0).values.reshape(20, 20)

# Muestra la imagen promedio en escala de grises (zonas claras = píxeles que suelen encenderse).
plt.imshow(mean_image, cmap="gray")
# Agrega título al gráfico actual.
plt.title("Imagen promedio")
# Renderiza el gráfico en pantalla.
plt.show()

# Muestra el mapa de desviación estándar (color más claro = más variación entre imágenes, blanco siendo mas variable, negro siendo menos variable, pasando por rojo y amarillo).
plt.imshow(std_image, cmap="hot")
# Agrega título al gráfico actual.
plt.title("Desviación estándar por píxel")
# Renderiza el gráfico en pantalla.
plt.show()

#==========================================
# Viendo la matriz de desviacion estandard  elegimos los siguientes atributos. 

def k_mayores_matriz_con_indices(matriz, k):
    # Creamos una lista con (valor, fila, columna)
    valores_con_indices = []
    for i in range(len(matriz)):
        for j in range(len(matriz[i])):
            valores_con_indices.append([matriz[i][j], i, j])

#Ordenamos de mayor a menor usando sort normal
    valores_con_indices.sort(reverse=True)  # ordena por el primer elemento de cada sublista

#Tomamos los k primeros
    top_k = valores_con_indices[:k]

# Nos quedamos con los indices
    indices = [(v[1], v[2]) for v in top_k]

    return indices

indices = k_mayores_matriz_con_indices(desviacion_estandard, 9)

def posicion_en_df (indices):
    atributos = []
    for posicion in indices:
        if posicion[0] == 0:
            atributos.append(posicion[1])
        elif posicion[0] == 1:
            atributos.append((posicion[1]+20))
        else:
            atributos.append(((posicion[0] - 1) * 20) + posicion[1])
    return atributos

indices = posicion_en_df(indices)

# Elegimos nuestros atributos, los usamos.

#Dividimos en tres los atributos más revelantes, como pide la consigna.
def dividir(seq, n):
    atributos = []
    for i in range(0, len(seq), n):
        atributos.append(seq[i:i+n])
    return atributos

atributos = dividir(indices, 3)

# Tengo mis atributos
atributos_1 = atributos[0]
atributos_2 = atributos[1]
atributos_3 = atributos[2]

"""
def evaluar_knn(columnas, k):
    # Entrenamos con esas columnas
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train[columnas], y_train)
    y_pred = knn.predict(X_test[columnas])
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=5)
    
    print(f"Atributos: {columnas} | k={k} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print("------")
"""
def evaluar_knn(columnas_idx, k):
    # aseguramos enteros válidos
    columnas_idx = list(map(int, columnas_idx))

    # seleccionar por POSICIÓN
    Xtr = X_train.iloc[:, columnas_idx]
    Xte = X_test.iloc[:, columnas_idx]

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtr, y_train)
    y_pred = knn.predict(Xte)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, pos_label=5)
    
    print(f"Atributos (posiciones): {columnas_idx} | k={k} | "
          f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(confusion_matrix(y_test, y_pred, labels=[4,5]))
    print("------")
    
# Probamos con diferentes grupos de atributos
evaluar_knn(atributos_1, k=3)
evaluar_knn(atributos_2, k=3)
evaluar_knn(atributos_3, k=3)
#%%
# 2d) probamos para los subjonjuntos de atributos elegidos, 
# las predicciones con k distintos.

evaluar_knn(atributos_1, k=3)
evaluar_knn(atributos_2, k=3)
evaluar_knn(atributos_3, k=3)
evaluar_knn(atributos_1, k=10)
evaluar_knn(atributos_2, k=10)
evaluar_knn(atributos_3, k=10)
evaluar_knn(atributos_1, k=1)
evaluar_knn(atributos_2, k=1)
evaluar_knn(atributos_3, k=1)