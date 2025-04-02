"""Calculaaremos las distancias entre todos los pares de puntos y 
determinaremos cuales estan mas alegados entre si y cuales estan mas cercanos utilizando las distancias
euclidianas, Manhattan, Chebyshev"""
import numpy as np
import pandas as pd
from scipy.spatial import distance

# Determinamos las coordenadas de las tiendas
puntos = {
    'Punto A':(2,3),
    'Punto B':(5,4),
    'Punto C':(1,1),
    'Punto D':(6,7),
    'Punto E':(3,5),
    'Punto F':(8,2),
    'Punto G':(4,6),
    'Punto H':(2,1)
}

# Convertir las coordenadas a un dataframe
df_puntos = pd.DataFrame(puntos).T
df_puntos.columns = ['X', 'Y']
print('===========Coordenadas de los Puntos:===========\n')
print(df_puntos)

def calcular_distancias_euclidean(puntos):
    distancias = pd.DataFrame(index=df_puntos.index, columns=df_puntos.index)
    for i in df_puntos.index:
        for j in df_puntos.index:
            if i != j:
                distancias.loc[i, j] = distance.euclidean(df_puntos.loc[i], df_puntos.loc[j])
    return distancias

def calcular_distancias_manhattan(puntos):
    distancias = pd.DataFrame(index=df_puntos.index, columns=df_puntos.index)
    for i in df_puntos.index:
        for j in df_puntos.index:
            if i != j:
                distancias.loc[i, j] = distance.cityblock(df_puntos.loc[i], df_puntos.loc[j])
    return distancias

def calcular_distancias_chebyshev(puntos):
    distancias = pd.DataFrame(index=df_puntos.index, columns=df_puntos.index)
    for i in df_puntos.index:
        for j in df_puntos.index:
            if i != j:
                distancias.loc[i, j] = distance.chebyshev(df_puntos.loc[i], df_puntos.loc[j])
    return distancias

# Calcular las distancias
distancias_euclidean = calcular_distancias_euclidean(puntos)
distancias_manhattan = calcular_distancias_manhattan(puntos)
distancias_chebyshev = calcular_distancias_chebyshev(puntos)

print('\n===========Distancias Euclidianas===========')
print(distancias_euclidean)
valor_maximo = distancias_euclidean.values.max()
(punto1, punto2) = distancias_euclidean.stack().idxmax()
print('Distancia máxima euclidiana:', valor_maximo)
print('Entre el punto:', punto1, 'y el punto:', punto2)

print('\n===========Distancias Manhattan===========')
print(distancias_manhattan)
valor_maximo = distancias_manhattan.values.max()
(punto1, punto2) = distancias_manhattan.stack().idxmax()
print('Distancia máxima Manhattan:', valor_maximo)
print('Entre el punto:', punto1, 'y el punto:', punto2)

print('\n===========Distancias Chebyshev===========')
print(distancias_chebyshev)
valor_maximo = distancias_chebyshev.values.max()
(punto1, punto2) = distancias_chebyshev.stack().idxmax()
print('Distancia máxima Chebyshev:', valor_maximo)
print('Entre el punto:', punto1, 'y el punto:', punto2)
#Otra manera
max_value=distancias_euclidean.max().max()
col_max = distancias_euclidean.max().idxmax()
id_max = distancias_euclidean[col_max].idxmax()
print(f'Valor maximo:{max_value}')
print(f'Columna:{col_max}')
print(f'Indice:{id_max}')