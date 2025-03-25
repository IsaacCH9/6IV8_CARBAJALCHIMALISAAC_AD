import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos, se separan por ; se tiene que especificar
df = pd.read_csv('proyecto1.csv', sep=';')  # Archivo principal
df2 = pd.read_csv('Catalogo_sucursal.csv', sep=';')  # Catálogo de sucursales

# Ventas totales del comercio
ventas_totales = df['ventas_tot'].sum()
print(f"Ventas totales del comercio: {ventas_totales:,.2f}")

# Socios con adeudo y sin adeudo con su porcentaje correspondiente
socios_con_adeudo = df[df['B_adeudo'] == 'Con adeudo']
socios_sin_adeudo = df[df['B_adeudo'] == 'Sin adeudo']
total_socios = len(socios_con_adeudo) + len(socios_sin_adeudo)
porcentaje_con_adeudo = (len(socios_con_adeudo) / total_socios) * 100
porcentaje_sin_adeudo = (len(socios_sin_adeudo) / total_socios) * 100
print(f"Socios con adeudo: {len(socios_con_adeudo)} ({porcentaje_con_adeudo:.2f}%)")
print(f"Socios sin adeudo: {len(socios_sin_adeudo)} ({porcentaje_sin_adeudo:.2f}%)")

# Gráfica de ventas totales respecto del tiempo en gráfica de barras
plt.figure(figsize=(10, 6))
plt.bar(df['B_mes'], df['ventas_tot'], color='blue')
plt.title('Ventas Totales con el Tiempo')
plt.xlabel('Mes')
plt.ylabel('Ventas Totales')
plt.xticks(rotation=45)
plt.show()

# La columna usa float
df['pagos_tot'] = df['pagos_tot'].str.replace(',', '.').astype(float)

# Gráfica de desviación estándar de los pagos realizados respecto del tiempo
desviacion_estandar = df.groupby('B_mes')['pagos_tot'].std()
plt.figure(figsize=(10, 6))
plt.plot(desviacion_estandar, marker='o', color='red')
plt.title('Desviación Estándar de Pagos Realizados Respecto del Tiempo')
plt.xlabel('Mes')
plt.ylabel('Desviación Estándar')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Deuda total de los clientes
# La columna usa float
df['adeudo_actual'] = df['adeudo_actual'].str.replace(',', '.').astype(float)
deuda_total = df['adeudo_actual'].sum()
print(f"Deuda total de los clientes: {deuda_total:,.2f}")

# Porcentaje de utilidad del comercio
utilidad = ventas_totales - deuda_total
porcentaje_utilidad = (utilidad / ventas_totales) * 100
print(f"Porcentaje de utilidad del comercio: {porcentaje_utilidad:.2f}%")

# Gráfico circular
ventas_por_sucursal = df.groupby('id_sucursal')['ventas_tot'].sum().reset_index()
ventas_por_sucursal = ventas_por_sucursal.merge(df2, on='id_sucursal', how='left')
plt.figure(figsize=(8, 8))
plt.pie(ventas_por_sucursal['ventas_tot'], labels=ventas_por_sucursal['suc'], autopct='%1.1f%%', startangle=90)
plt.title('Ventas por Sucursal')
plt.show()

# Gráfico de deudas totales por sucursal respecto del margen de utilidad
deudas_por_sucursal = df.groupby('id_sucursal')['adeudo_actual'].sum().reset_index()
deudas_por_sucursal = deudas_por_sucursal.merge(df2, on='id_sucursal', how='left')

# Calcular el margen de utilidad por sucursal
utilidad_por_sucursal = ventas_por_sucursal['ventas_tot'] - deudas_por_sucursal['adeudo_actual']

#Crear la grafica de barras
plt.figure(figsize=(10, 6))
x = np.arange(len(deudas_por_sucursal['suc']))
plt.bar(x - 0.2, deudas_por_sucursal['adeudo_actual'], width=0.4, label='Deudas Totales', color='orange')
plt.bar(x + 0.2, utilidad_por_sucursal, width=0.4, label='Margen de Utilidad', color='green')
plt.xticks(x, deudas_por_sucursal['suc'], rotation=45)
plt.title('Deudas Totales y Margen de Utilidad por Sucursal')
plt.xlabel('Sucursal')
plt.ylabel('Monto')
plt.legend()
plt.tight_layout()
plt.show()