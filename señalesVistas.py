import csv
import matplotlib.pyplot as plt

# Ruta del archivo CSV (ajusta si es necesario)
archivo_csv = "senal_v2.csv"

# Leer la señal desde la primera fila del archivo
with open(archivo_csv, 'r') as f:
    reader = csv.reader(f)
    filas = list(reader)
    if filas:
        datos_senal = list(map(float, filas[-1]))  # Última fila guardada
    else:
        raise ValueError("El archivo CSV está vacío")

# Crear gráfico
plt.figure(figsize=(12, 4))
plt.plot(datos_senal, label="Señal registrada", color="blue")
plt.title("Visualización de la señal")
plt.xlabel("Muestra")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()