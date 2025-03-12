import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar datos
df = pd.read_csv('housing.csv')

# Calcular estadísticas descriptivas con Pandas
stats_pandas = df["median_house_value"].agg(["mean", "median", "var", "std"]).to_frame()

# Agregar moda y rango manualmente
stats_pandas.loc["Moda"] = df["median_house_value"].mode().values[0]
stats_pandas.loc["Rango"] = np.ptp(df["median_house_value"])

# Renombrar índices para mayor claridad
stats_pandas.index = ["Media", "Mediana", "Varianza", "Desviación Estándar", "Moda", "Rango"]

# Convertir a DataFrame para mostrar
stats_pandas_df = stats_pandas.reset_index()
stats_pandas_df.columns = ["Estadística", "Valor"]

# Crear tabla de frecuencias con intervalos de 50,000 en 50,000
bin_edges = range(int(df["median_house_value"].min()), int(df["median_house_value"].max()) + 50000, 50000)
df["intervalo"] = pd.cut(df["median_house_value"], bins=bin_edges, right=False)
freq_table_pandas = df["intervalo"].value_counts().reset_index()
freq_table_pandas.columns = ["Intervalo de Valores", "Frecuencia"]
freq_table_pandas = freq_table_pandas.sort_values(by="Intervalo de Valores")

# Mostrar estadísticas descriptivas en formato tabla
print("\n================ ESTADÍSTICAS DESCRIPTIVAS ================".center(80))
print(stats_pandas_df.to_string(index=False))
print("\n================= TABLA DE FRECUENCIAS =================".center(80))
print(freq_table_pandas.to_string(index=False))

# Configurar la figura y los subgráficos
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 fila, 3 columnas

columns_to_plot = ["median_house_value", "total_bedrooms", "population"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Azul, Naranja, Verde

for i, column in enumerate(columns_to_plot):
    axes[i].hist(df[column].dropna(), bins=30, alpha=0.7, color=colors[i])
    axes[i].set_title(f"Histograma de {column.replace('_', ' ').title()}")
    axes[i].set_xlabel("Valor")
    axes[i].set_ylabel("Frecuencia")
    axes[i].grid(True)

# Ajustar espacio entre los gráficos
plt.tight_layout()
plt.show()
