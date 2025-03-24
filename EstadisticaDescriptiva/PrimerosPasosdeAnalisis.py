import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('proyecto1.csv')  # Archivo principal
df2 = pd.read_csv('Catalogo_sucursal.csv')  # Catálogo de sucursales

# Ventas totales del comercio
ventas_totales = df['ventas_tot'].sum()
print(f"Ventas totales del comercio: {ventas_totales}")

# Socios con adeudo y sin adeudo con su porcentaje
socios_con_adeudo = df[df['B_adeudo'] == 'Con adeudo']
socios_sin_adeudo = df[df['B_adeudo'] == 'Sin adeudo']
print(f"Socios con adeudo: {socios_con_adeudo}")
print(f"Socios sin adeudo: {socios_sin_adeudo}")
total_socios = socios_con_adeudo + socios_sin_adeudo
porcentaje_con_adeudo = (socios_con_adeudo / total_socios) * 100
porcentaje_sin_adeudo = (socios_sin_adeudo / total_socios) * 100
print(f'Porcentaje de socios con adeudo: {porcentaje_con_adeudo}%')
print(f'Porcentaje de socios sin adeudo: {porcentaje_sin_adeudo}%')

# Graficar las ventas totales respecto del tiempo en grafica de barras
plt.figure(figsize=(10, 6))
plt.bar(df['B_mes'], df['ventas_tot'], color='blue')
plt.title('Ventas Totales Respecto del Tiempo')
plt.xlabel('Mes')
plt.ylabel('Ventas Totales')
plt.show()

# Grafica de desviacion estandar respecto del tiempo
desviacion_estandar = df.groupby('B_mes')['pagos_realizados'].std()
plt.figure(figsize=(10, 6))
plt.plot(desviacion_estandar, marker='o', color='red')
plt.title('Desviación Estándar de Pagos Realizados Respecto del Tiempo')
plt.xlabel('Mes')
plt.ylabel('Desviación Estándar')
plt.grid()
plt.show()

#Deuda total de los socios