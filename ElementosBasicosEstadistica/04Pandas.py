import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('housing.csv')

#mostrar las primeras 5 filas.
print(df.head())

#mostrar las ultimas 5 filas
print(df.tail())

#mostrar filas en especifico
print(df.iloc[7])

#mostrar columna oceans_proximity
print(df['ocean_proximity'])

#obtener la media de la columna total_rooms
mediadecuarto = df['total_rooms'].mean()
print('La media de total room es: ' , mediadecuarto)

#mediana
medianacuarto = df['median_house_value'].median()
print('La mediana de la columna valor mediana de la casa : ' , medianacuarto)

#la suma de popular
salariototal = df['population'].sum()
print('el salario total es: ' , salariototal)

#par poder filtrar
vamoshacerunfiltro = df[df['ocean_proximity'] == 'ISLAND']
print(vamoshacerunfiltro)

#Vamos hacer un grafico de dispercion
plt.scatter(df['ocean_proximity'][:10],df['median_house_value'][:10])
#nombramos los ejes
plt.xlabel('ocean_proximity')
plt.ylabel('median_house_value')
plt.title('Grafico de dispersion de Proximidad al Oceano vs Precio')
plt.show()
