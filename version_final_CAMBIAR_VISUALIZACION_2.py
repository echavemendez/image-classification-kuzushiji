
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
# Imports necesarios.

# Para manejo de datasets
import duckdb as dd
import pandas as pd
import os
# Para calculos y operaciones numericas
import numpy as np
# Visualizaciones
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
# Para modelos y machine learning en general
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# Metricas y comparaciones
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,  precision_score, recall_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
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

# ANÁLISIS EXPLORATORIO DE DATOS 
# En esta sección analizamos la estructura general del dataset,
# su distribución, balance de clases y algunas visualizaciones básicas
# que nos ayuden a entender mejor la información antes de modelar.


# Vistazo de la estructura general del dataset
# Empezamos mirando la estructura básica para tener una idea del contenido.
print("Dimensiones del dataset de imágenes:", df_imagenes.shape)
print("\nPrimeras filas del dataset:")
print(df_imagenes.head())

#%% 
# Vemos los tipos de datos y si hay valores faltantes
print("\nInformación general de df_imagenes:")
df_imagenes.info()
print("\nValores faltantes por columna:")
print(df_imagenes.isnull().sum().sum())  # Total de valores faltantes en el dataset

# Notamos que todos los atributos tienen valores no nulos
#%%
# Ahora nos vamos a meter un poco mas profundo en el analisis de las variables
# Contamos cuántas muestras hay de cada etiqueta (0 a 9)

cantidad_de_muestras = dd.sql("""
                SELECT label, COUNT(*) AS apariciones
                FROM df_labels
                GROUP BY label
                ORDER BY label
                """).df()

promedio = cantidad_de_muestras['apariciones'].sum() / 10

#%% 
# Graficamos los resultados

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
# Visualizaciones:
# Para hacernos una idea visual de cómo son los caracteres,
# vamos a tomar un representante de cada clase para ver a grandes rasgos como son.
# Creamos el mapeo de clase → carácter japonés

# Índices de las filas que queremos mostrar
indices = [2, 3, 5, 98, 225, 329, 320, 411, 406, 407]

# Clases correspondientes a cada imagen
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  

# Creamos la figura y subplots
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

# Recorremos las imágenes y las mostramos haciendo uso de la funcion dada en el enunciado del tp con un forloop

for i, ax in enumerate(axes.flat):
    idx = indices[i]
    img = np.array(df_imagenes.iloc[idx]).reshape((28, 28))
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Clase {labels[i]}", fontsize=10)
    ax.axis('off')  

# Ajustamos espacios para que no se solapen los títulos

plt.suptitle("Ejemplos representativos de cada clase en el dataframe", fontsize=14, y=1.03)
plt.tight_layout()
plt.show()

#%%
# También compararemos imagenes con misma etiqueta para entender que tanto difieren, elegimos seguir la sugerencia de la catedra (ejemplo: clase 8)

# Filtramos las filas que corresponden a la clase 8

df_clase8 = dd.sql("""
    SELECT *
    FROM df_imagenes_labels
    WHERE label = 8
""").df()

# Tomamos algunos índices aleatorios de esa clase
muestras = df_clase8.sample(40, random_state=42)  # elegimos 40 imágenes al azar
muestras_sin_label = muestras.iloc[:, :-1].copy() 
clase = 8

# Graficamos las imágenes seleccionadas
fig, axes = plt.subplots(5, 8, figsize=(10, 6))

for i, ax in enumerate(axes.flat):
    img = np.array(muestras_sin_label.iloc[i]).reshape((28, 28))
    ax.imshow(img, cmap='gray')
    ax.axis('off')

plt.suptitle(f"Comparación de imágenes de la clase {clase}", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

#%% 
# Ordenamos el dataset para trabajar mas comodos

df_imagenes_labels_ord = dd.sql("""
                                SELECT *
                                FROM df_imagenes_labels 
                                ORDER BY label
                                """).df()
                
#%% 
# Calculamos la imagen promedio para cada clase

m_prom_clase = [np.zeros((28,28)) for _ in range(10)] # Creamos 10 matrices inicializadas en 0, para luego rellenarlas

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
plt.suptitle("Similitud promedio de imagenes clase 1 - clase2", fontsize=14, y=1.02)
ax.axis('off')

fig, ax = plt.subplots()
ax.imshow(diferencia_6_2,cmap='gray')
plt.suptitle("Similitud promedio de imagenes clase 6 - clase2", fontsize=14, y=1.02)
ax.axis('off')

#%%
# Calculamos imagen promedio de la clase 8 especificamente, para referenciarla a posteriori en el informe.

matriz_8= m_prom_clase[8]
fig, ax = plt.subplots()
ax.imshow(matriz_8,cmap='gray')
plt.suptitle("Imagen promedio de la clase 8", fontsize=14, y=1.02)
ax.axis('off')

#%% 
# Variabilidad de los pixeles entre las imagenes

# En este gráfico mostramos la variabilidad entre los pixeles de todas las imagenes
# Calcula la intensidad promedio de CADA píxel (columna) a lo largo de TODAS las imágenes (filas),
# reacomodamos los pixeles en una matriz de 28x28 para visualizar como imagen.

matriz_prom = df_imagenes.mean(axis=0).values.reshape(28, 28)

# Calcula el desvío estándar (variabilidad) de CADA píxel entre imágenes,
# lo pasa a array y lo reacomoda a 28x28 para graficar.
std_image = df_imagenes.std(axis=0).values.reshape(28, 28)

# Muestra la imagen promedio en escala de grises (zonas claras = píxeles que suelen encenderse).
plt.imshow(matriz_prom, cmap="gray")
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
# Nos armamos una lista con los indices que vamos a necesitar borrar para descartar los datos.
# Son los 4 primeros y los 4 ultimos de cada fila.
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

# eliminamos esas columnas del DataFrame que nos dieron por consigna

df_datos_calidad = df_imagenes_labels.drop(df_imagenes_labels.columns[columnas_a_descartar], axis=1)
#%%

# Visualizacion de nuestro nuevo df
# Para garantizar que efectivamente hubo una mejora
matriz_filtrada2 = matriz_prom[4:-4, 4:-4]
fig, ax = plt.subplots()
ax.imshow(matriz_filtrada2, cmap='gray')
ax.axis('off')
#%%

# Comenzamos con el trabajo sobre el modelo de clasificacion binaria (punto 2)
# 2. a)

# Armamos un dataframe solamente con las filas de clases 4 y 5 

query_cuatro_y_cinco =  """
SELECT *
FROM df_datos_calidad 
WHERE (label = 4 
       OR
       label = 5)

"""

df_cuatro_y_cinco = dd.query(query_cuatro_y_cinco).df()
    
# Sabemos que esta balanceado porque previamente lo calculamos.

#%%
#2b) Separamos datos de prueba y datos de entrenamiento

# Separamos los atributos (X) y las etiquetas (y)

X = df_cuatro_y_cinco.drop(columns='label')
y = df_cuatro_y_cinco['label']

# 80% para entrenamiento y 20% para test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # proporción de test
    random_state=42,     
    stratify=y           # para preservar balance
)

print("Tamaño train:", X_train.shape[0])
print("Tamaño test:", X_test.shape[0])
print("\nDistribución de clases en train:")
print(y_train.value_counts(normalize=True))
print("\nDistribución de clases en test:")
print(y_test.value_counts(normalize=True))

#%%
#2c)

# Elegimos los atributos para llevar a cabo nuestro modelo knn y comparar 

# Variabilidad de pixeles de imagen a imagen

# En este gráfico mostramos la variabilidad entre los pixeles de todas las clases 4 y 5.
# Repetimos el grafico de desviacion por pixel para elegir conjuntos de 3 mas relevantes.
 
df_sin_ultima = df_cuatro_y_cinco.drop(df_cuatro_y_cinco.columns[-1], axis=1)
desviacion_estandard = df_sin_ultima.std(axis=0).values.reshape(20, 20)

# Calcula la intensidad promedio de CADA píxel (columna) a lo largo de TODAS las imágenes (filas),
# y lo reacomoda a una matriz de 28x28 para visualizar como imagen.

matriz_prom_calidad = df_sin_ultima.mean(axis=0).values.reshape(20, 20) # matriz promedio sobre nuestro df con clases 4 y 5

# Calcula el desvío estándar (variabilidad) de CADA píxel entre imágenes,
# lo pasa a array y lo reacomoda a 28x28 para graficar.
std_image = df_sin_ultima.std(axis=0).values.reshape(20, 20)

# Muestra la imagen promedio en escala de grises (zonas claras = píxeles que suelen encenderse).
plt.imshow(matriz_prom_calidad, cmap="gray")
# Agrega título al gráfico actual.
plt.title("Imagen promedio")
# Renderiza el gráfico en pantalla.
plt.show()

# Muestra el mapa de desviación estándar (color más claro = más variación entre imágenes, 
# blanco siendo mas variable, negro siendo menos variable, pasando por rojo y amarillo).
plt.imshow(std_image, cmap="hot")
# Agrega título al gráfico actual.
plt.title("Desviación estándar por píxel")
# Renderiza el gráfico en pantalla.
plt.show()

#%%
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

# Armamos nuestror arrays con los conjuntos de datos que luego "partiremos" y utilizaremos para nuesto modelo.
pares_indices_1 = k_mayores_matriz_con_indices(desviacion_estandard, 9)
pares_indices_2 = k_mayores_matriz_con_indices(desviacion_estandard, 15)
pares_indices_3 = k_mayores_matriz_con_indices(desviacion_estandard, 30)
pares_indices_4 = k_mayores_matriz_con_indices(desviacion_estandard, 3)

def posicion_en_df(indices):
    # n_columnas = ancho de la matriz (por ejemplo 20)
    atributos = [fila * 20 + columna for fila, columna in indices]
    return atributos

indices_1 = posicion_en_df(pares_indices_1)
indices_2 = posicion_en_df(pares_indices_2)
indices_3 = posicion_en_df(pares_indices_3)
indices_4 = posicion_en_df(pares_indices_4)

# Elegimos nuestros atributos, los usamos.

# Hacemos una función que divida una lista en sublistas de tamaño n. Nos sirve para separar nuestros indices 
# previamente calculados

def dividir(seq, n):
    atributos = []
    for i in range(0, len(seq), n):
        atributos.append(seq[i:i+n])
    return atributos

# Usamos nuestra función y tenemos las posiciones de los atributos que usaremos luego.

atributos_de_a_3 = dividir(indices_1, 3)
atributos_de_a_5 = dividir(indices_2, 5)
atributos_de_a_10 = dividir(indices_3, 10)
atributos_de_a_1 = dividir(indices_4, 1)

# Hacemos nuestro modelo KNN
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
    matriz = confusion_matrix(y_test, y_pred, labels=[4,5])
    
    return {
        "atributos": columnas_idx,
        "k": k,
        "accuracy": round(acc, 4),
        "matriz_de_confusion": matriz
    }

#%%
# Probamos con estos grupos con la k = 3

# Separamos en nuestros atributos en conjuntos de 3.
atributos_1 = atributos_de_a_3[0]
atributos_2 = atributos_de_a_3[1]
atributos_3 = atributos_de_a_3[2]

# Evaluamos los conjuntos de 3 en el modelo. 
atributos1_k3 = evaluar_knn(atributos_1, k=3)
atributos2_k3 = evaluar_knn(atributos_2, k=3)
atributos3_k3 = evaluar_knn(atributos_3, k=3)

# Resultados de el modelo. Accuracy y Matriz de Confusión.
resultados_3 = [
    atributos1_k3,
    atributos2_k3,
    atributos3_k3
]
df_3_atributos_k3 = pd.DataFrame(resultados_3)[["atributos", "k", "accuracy", "matriz_de_confusion"]]

# Ahora probamos con 3 grupos de 5 atributos cada uno. 
atributos_4 = atributos_de_a_5[0]
atributos_5 = atributos_de_a_5[1]
atributos_6 = atributos_de_a_5[2]

# Evaluamos los conjuntos de 5 en el modelo.
atributos4_k3 = evaluar_knn(atributos_4, k=3)
atributos5_k3 = evaluar_knn(atributos_5, k=3)
atributos6_k3 = evaluar_knn(atributos_6, k=3)

# Resultados de el modelo. Accuracy y Matriz de Confusión.
resultados_5 = [
    atributos4_k3,
    atributos5_k3,
    atributos6_k3
]
df_5_atributos_k3 = pd.DataFrame(resultados_5)[["atributos", "k", "accuracy", "matriz_de_confusion"]]

#Ahora probamos con 3 grupos de 10 atributos cada uno.
atributos_7 = atributos_de_a_10[0]
atributos_8 = atributos_de_a_10[1]
atributos_9 = atributos_de_a_10[2]

# Evaluamos los conjuntos de 10 en el modelo.
aributos7_k3 = evaluar_knn(atributos_7, k=3)
aributos8_k3 = evaluar_knn(atributos_8, k=3)
aributos9_k3 = evaluar_knn(atributos_9, k=3)

# Resultados de el modelo. Accuracy y Matriz de Confusión.
resultados_10 = [
    aributos7_k3,
    aributos8_k3,
    aributos9_k3
]
df_10_atributos_k3 = pd.DataFrame(resultados_10)[["atributos", "k", "accuracy", "matriz_de_confusion"]]

#Ahora probamos con 3 grupos de 1 atributo cada uno.
atributos_10 = atributos_de_a_1[0]
atributos_11 = atributos_de_a_1[1]
atributos_12 = atributos_de_a_1[2]

# Evaluamos los conjuntos de 1 en el modelo.
aributos10_k3 = evaluar_knn(atributos_10, k=3)
aributos11_k3 = evaluar_knn(atributos_11, k=3)
aributos12_k3 = evaluar_knn(atributos_12, k=3)

# Resultados de el modelo. Accuracy y Matriz de Confusión.
resultados_1 = [
    aributos10_k3,
    aributos11_k3,
    aributos12_k3
]
df_1_atributos_k3 = pd.DataFrame(resultados_1)[["atributos", "k", "accuracy", "matriz_de_confusion"]]

#ordenamos las tablas segun k descendiente y segun accuracy descendiente
df_3_atributos_k3 = df_3_atributos_k3.sort_values(by=['k', 'accuracy'], ascending=[False, False]).reset_index(drop=True)
df_1_atributos_k3 = df_1_atributos_k3.sort_values(by=['k', 'accuracy'], ascending=[False, False]).reset_index(drop=True)
df_10_atributos_k3 = df_10_atributos_k3.sort_values(by=['k', 'accuracy'], ascending=[False, False]).reset_index(drop=True)
df_5_atributos_k3 = df_5_atributos_k3.sort_values(by=['k', 'accuracy'], ascending=[False, False]).reset_index(drop=True)

#================================================= VISUALIZACIÓN ==========================================================

from matplotlib.colors import LinearSegmentedColormap

matriz_confusion_mejor = df_10_atributos_k3.iloc[-1, -1]
cm = matriz_confusion_mejor

cmap_azules = LinearSegmentedColormap.from_list('azules', ['#C6DBEF', '#08306B'], N=256)

fig, ax = plt.subplots(figsize=(4.5,4))
im = ax.imshow(cm, aspect='equal', cmap=cmap_azules)

for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, f'{val}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# 🔒 Quitar todo lo de los ejes
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([]); ax.set_yticks([])
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title('Matriz de Confusión')
plt.tight_layout()
plt.show()

#%%
#2. d. Probamos con distintos k y los mismos atributos que la 2.c, separando cada dataframe por el valor de k.
# Probamos el modelo con los mismos atributos que antes, pero cambiando el k por k=10, k=30 y k=2
# Guardamos todos los resultados en listas limpias

#Lista de resultados con 3 conjuntos de un solo atributo
resultados_1 = [
    evaluar_knn(atributos_10, k=2),
    evaluar_knn(atributos_10, k=10),
    evaluar_knn(atributos_10, k=30),
    evaluar_knn(atributos_11, k=2),
    evaluar_knn(atributos_11, k=10),
    evaluar_knn(atributos_11, k=30),
    evaluar_knn(atributos_12, k=2),
    evaluar_knn(atributos_12, k=10),
    evaluar_knn(atributos_12, k=30)
]
df_atributos1 = pd.DataFrame(resultados_1)

#Lista de resultados con 3 conjuntos de 5 atributos
resultados_5 = [
    evaluar_knn(atributos_4, k=2),
    evaluar_knn(atributos_4, k=10),
    evaluar_knn(atributos_4, k=30),
    evaluar_knn(atributos_5, k=2),
    evaluar_knn(atributos_5, k=10),
    evaluar_knn(atributos_5, k=30),
    evaluar_knn(atributos_6, k=2),
    evaluar_knn(atributos_6, k=10),
    evaluar_knn(atributos_6, k=30)
]
df_atributos5 = pd.DataFrame(resultados_5)

#Lista de resultados con 3 conjuntos de 10 atributos
resultados_10 = [
    evaluar_knn(atributos_7, k=2),
    evaluar_knn(atributos_7, k=10),
    evaluar_knn(atributos_7, k=30),
    evaluar_knn(atributos_8, k=2),
    evaluar_knn(atributos_8, k=10),
    evaluar_knn(atributos_8, k=30),
    evaluar_knn(atributos_9, k=2),
    evaluar_knn(atributos_9, k=10),
    evaluar_knn(atributos_9, k=30)
]
df_atributos10 = pd.DataFrame(resultados_10)
#============================================== VISUALIZACIÓN =================================================0
# Hacemos una visualización para poder observar como se comportan los conjuntos de atributos de 1, 5 y 10
# con los distintos k respectivamente 
# CAMBIAR VISUALIZACIÓN

# Agrego una columna en cada DataFrame con los atributos que tienen cada uno.
df_atributos1["num_atributos"] = 1
df_atributos5["num_atributos"] = 5
df_atributos10["num_atributos"] = 10

# Hago un DataFrame concatenando todo
df_todos = pd.concat([df_atributos1, df_atributos5, df_atributos10], ignore_index=True)

pivot = df_todos.pivot_table(
    index="num_atributos", columns="k", values="accuracy"
)


# Gráfico de barras de 3 valores distintos de k, y de tres grupos de atributos distintos
palette = ["#B5E2FA", "#64B5F6", "#1976D2"]

plt.figure(figsize=(10,6))
ax = sns.barplot(
    data=df_todos,
    x="k", 
    y="accuracy", 
    hue="num_atributos", 
    palette=palette,
    errorbar=None  # sin líneas verticales
)

# Mostrar el valor numérico de cada barra y los atributos
for i, p in enumerate(ax.patches):
    height = p.get_height()
    atributo = df_todos.iloc[i]["atributos"]
    ax.text(
        p.get_x() + p.get_width()/2, height + 0.002,
        f"{height:.3f}\n{atributo}",
        ha="center", va="bottom", fontsize=8, rotation=90
    )

plt.title("Accuracy según cantidad de atributos y valor de k", fontsize=14)
plt.xlabel("Valor de k")
plt.ylabel("Accuracy")
plt.ylim(0.5, 0.9)  # hace zoom en la parte superior
plt.legend(title="N° de atributos")
plt.tight_layout()
plt.show()

#%%
#evaluamos con k = 100 para corroborar una estabilidad del modelo.
print(evaluar_knn(atributos_10, k = 100)) #con un atributo
print(evaluar_knn(atributos_12, k=3))
print(evaluar_knn(atributos_11, k=3))
#con 10 atributos
print(evaluar_knn(atributos_8, k=3))
print(evaluar_knn(atributos_9, k=3))
#print(evaluar_knn(atributos_7, k=100))

# los resultados de la exactitud son: [ 0.7682, 0.7957, 0.8161 ]

#%%
# Ejercicio 3
  

X_m = df_datos_calidad.drop(columns='label') # variable sobre la cual entreno
y_m = df_datos_calidad['label'] # etiquetas para comparacion

# Separamos training set y el validation set (nuestro held-out)

# 80% para entrenamiento y 20% para test
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_m, y_m,
    test_size=0.2,       # proporción de test
    random_state=42,     # para reproducibilidad
    stratify=y_m           # mantiene el balance entre clases, buscamos que se mantenga ya que esta perfectamente balanceado como nos lo dan.
)
#%%
# En (b) probamos distintas profundidades y medimos la accuracy promedio.

# Ajustamos árboles de decisión con distintas profundidades

profundidades = range(1, 11)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

resultados = []

for d in profundidades:
    arbol = DecisionTreeClassifier(max_depth=d, random_state=42)
    puntajes = cross_val_score(arbol, X_m_train, y_m_train, cv=cv, scoring='accuracy')
    media = puntajes.mean()
    resultados.append((d, media))
    print("Profundidad", d, "-> accuracy promedio:", round(media, 4))

# Pasamos los resultados a un DataFrame para analizar más fácil

df_resultados_m = pd.DataFrame(resultados, columns=['profundidad', 'accuracy_promedio'])

# Graficamos los resultados

plt.figure(figsize=(8,4))
plt.plot(df_resultados_m['profundidad'], df_resultados_m['accuracy_promedio'], marker='o')
plt.title("Accuracy promedio vs Profundidad del árbol")
plt.xlabel("Profundidad máxima")
plt.ylabel("Accuracy promedio (validación cruzada)")
plt.grid(alpha=0.3)
plt.show()

# Buscamos la mejor profundidad

mejor_fila = df_resultados_m.loc[df_resultados_m['accuracy_promedio'].idxmax()]
mejor_profundidad = mejor_fila['profundidad']

print("La profundidad mas optima encontrada:", mejor_profundidad)

print("Con un accuracy promedio de:", round(mejor_fila['accuracy_promedio'], 4))

#%%
# Probamos distintos criterios para el árbol con la mejor profundidad
# En (c) comparamos los criterios 'gini' y 'entropy', que son los vistos en clase, con la mejor profundidad.

criterios = ['gini', 'entropy']
mejor_accuracy = 0
mejor_configuracion = {}

print("Criterios:")

for crit in criterios:
    arbol = DecisionTreeClassifier(max_depth=int(mejor_profundidad), criterion=crit, random_state=42)
    puntajes = cross_val_score(arbol, X_m_train, y_m_train, cv=cv, scoring='accuracy')
    media = puntajes.mean()
    print("Criterio:", crit, "-> accuracy promedio:", round(media, 4))
    
    if media > mejor_accuracy:
        mejor_accuracy = media
        mejor_configuracion = {'max_depth': mejor_profundidad, 'criterion': crit}


print("La mejor combinación fue con profundidad:", mejor_configuracion['max_depth'], "y criterio =", mejor_configuracion['criterion'])
print("Accuracy promedio en cross validation:", round(mejor_accuracy, 4))

#%%
# Entrenamos el modelo final con TODO el conjunto de entrenamiento, ya que ya sabemos cual es el óptimo
# y lo evaluamos en el conjunto de validación (held-out)
# En (d) entrenamos ese mejor modelo final y evaluamos en el conjunto held-out.
mejor_arbol = DecisionTreeClassifier(
    max_depth=int(mejor_configuracion['max_depth']),
    criterion=mejor_configuracion['criterion'],
    random_state=42
)
mejor_arbol.fit(X_m_train, y_m_train)

# Predecimos sobre el conjunto de validación

y_pred = mejor_arbol.predict(X_m_test)

# Calculamos métricas

accuracy_final = accuracy_score(y_m_test, y_pred)

print("Accuracy final en el held-out:", round(accuracy_final, 4))

#%%
# Matriz de confusión

matriz_conf = confusion_matrix(y_m_test, y_pred)
etiquetas = np.unique(y_m_test)

plt.figure(figsize=(7, 5))

# Usamos un mapa de calor con una paleta clara y contraste medio
sns.heatmap(
    matriz_conf,
    annot=True,               # mostramos los valores dentro
    fmt='d',                  # formato entero
    cmap='Greys',             # escala de grises: ideal para imprimir
    cbar=False,               # sin barra lateral
    square=True,              # celdas cuadradas
    linewidths=0.5,           # líneas finas separadoras
    linecolor='black',        # mejora el contraste
    annot_kws={"size": 10}    # tamaño de fuente de los números
)

plt.title("Matriz de Confusión - Held-out", fontsize=12, pad=10)
plt.xlabel("Predicción", fontsize=10)
plt.ylabel("Verdadero", fontsize=10)
plt.xticks(ticks=np.arange(len(etiquetas)) + 0.5, labels=etiquetas, rotation=0)
plt.yticks(ticks=np.arange(len(etiquetas)) + 0.5, labels=etiquetas, rotation=0)
plt.tight_layout()
plt.show()
#%% 
# Métricas adicionales de clasificación multiclase

# Calculamos precisión, recall, f1 y accuracy

# Usamos las dos variantes principales: macro y micro, no usamos weighted porque ya esta balanceado entonces no añadiria informacion 

precision_macro = precision_score(y_m_test, y_pred, average='macro')
recall_macro = recall_score(y_m_test, y_pred, average='macro')
f1_macro = f1_score(y_m_test, y_pred, average='macro')

precision_micro = precision_score(y_m_test, y_pred, average='micro')
recall_micro = recall_score(y_m_test, y_pred, average='micro')
f1_micro = f1_score(y_m_test, y_pred, average='micro')


# Resultados para ver en terminal
print("Metricas adicionales para multiclase")
print(f"Precisión (macro):  {precision_macro:.4f}")
print(f"Recall (macro):     {recall_macro:.4f}")
print(f"F1-score (macro):   {f1_macro:.4f}")
print(f"\nPrecisión (micro):  {precision_micro:.4f}")
print(f"Recall (micro):     {recall_micro:.4f}")
print(f"F1-score (micro):   {f1_micro:.4f}")

# Armamos una tabla para referenciarla en el informe

column_labels = ['Métrica', 'Macro', 'Micro']
table_data = [
    ['Precisión', 0.7040, 0.7030],
    ['Recall',    0.7030, 0.7030],
    ['F1-score',  0.7027, 0.7030]
]

fig, ax = plt.subplots(figsize=(5, 1.5))
ax.axis('off')
tabla = ax.table(cellText=table_data, colLabels=column_labels, loc='center', cellLoc='center')

tabla.scale(1, 1.5)
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)

plt.title("Métricas de Clasificación Multiclase (macro y micro)", fontsize=11, pad=10)
plt.show()
