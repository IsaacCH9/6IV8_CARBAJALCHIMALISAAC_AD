import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('housing.csv')

stats_pandas = df["median_house_value"].agg(["mean", "median", "var", "std"]).to_frame()

stats_pandas.loc["Moda"] = df["median_house_value"].mode().values[0]
stats_pandas.loc["Rango"] = np.ptp(df["median_house_value"])

stats_pandas.index = ["Media", "Mediana", "Varianza", "Desviación Estándar", "Moda", "Rango"]

stats_pandas_df = stats_pandas.reset_index()
stats_pandas_df.columns = ["Estadística", "Valor"]

bin_edges = range(int(df["median_house_value"].min()), int(df["median_house_value"].max()) + 50000, 50000)
df["intervalo"] = pd.cut(df["median_house_value"], bins=bin_edges, right=False)
freq_table_pandas = df["intervalo"].value_counts().reset_index()
freq_table_pandas.columns = ["Intervalo de Valores", "Frecuencia"]
freq_table_pandas = freq_table_pandas.sort_values(by="Intervalo de Valores")


print("\nESTADÍSTICAS DESCRIPTIVAS".center(80))
print(stats_pandas_df.to_string(index=False))
print("\nTABLA DE FRECUENCIAS".center(80))
print(freq_table_pandas.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

columns_to_plot = ["median_house_value", "total_bedrooms", "population"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i, column in enumerate(columns_to_plot):
    axes[i].hist(df[column].dropna(), bins=25, alpha=0.8, color=colors[i])
    axes[i].set_title(f"Histograma de {column.replace('_', ' ').title()}")
    axes[i].set_xlabel("Valor")
    axes[i].set_ylabel("Frecuencia")
    axes[i].grid(True)

plt.tight_layout()
plt.show()