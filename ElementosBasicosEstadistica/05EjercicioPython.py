import pandas as pd
import matplotlib.pyplot as plt

housing = pd.read_csv('housing.csv')

def calcular_estadisticas(column):
    media = column.mean()
    mediana = column.median()
    moda = column.mode()[0]
    rango = column.max() - column.min()
    varianza = column.var()
    desviacion_std = column.std()
    
    return {
        "Media": media,
        "Mediana": mediana,
        "Moda": moda,
        "Rango": rango,
        "Varianza": varianza,
        "Desviación Estándar": desviacion_std
    }

def tabla_frecuencias(column):
    rango_min = column.min()
    rango_max = column.max()
    bins = range(int(rango_min), int(rango_max) + 500, 500)
    frecuencias = pd.cut(column, bins=bins).value_counts().reset_index()
    frecuencias.columns = ['Rango', 'Frecuencia']
    frecuencias = frecuencias.sort_values(by='Rango')  # Ordenar los rangos
    return frecuencias


resultados = calcular_estadisticas(housing['median_house_value'])
frecuencias = tabla_frecuencias(housing['median_house_value'])


print("Estadísticas Básicas:")
for key, value in resultados.items():
    print(f"{key}: {value:.2f}")

print("\nTabla de Frecuencias:")
print(frecuencias)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(housing['median_house_value'], bins=30, color='blue', alpha=0.7)
plt.title('Histogram: Median House Value')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 1)
plt.hist(housing['median_house_value'], bins=30, color='blue', alpha=0.7)
plt.title('Histogram: Median House Value')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
