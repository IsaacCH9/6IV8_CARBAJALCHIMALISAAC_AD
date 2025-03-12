import pandas as pd

def resumen_cotizacion(fichero):
    df = pd.read_csv(fichero, sep=';', decimal=',', thousands='.', index_col=0)
    
    resumen = pd.DataFrame({
        'Mínimo': df.min(),
        'Máximo': df.max(),
        'Media': df.mean(),
        'Desviación Estándar': df.std()
    })
    
    return resumen

resultado = resumen_cotizacion('cotizacion.csv')
print(resultado)
