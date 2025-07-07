import spidev
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Configurar SPI
spi = spidev.SpiDev()  # Crear un objeto SPI
spi.open(1, 2)         # Conectar al bus SPI 0, dispositivo CE0
spi.max_speed_hz = 1350000  # Configurar velocidad SPI (1.35 MHz)

# Función para leer un canal específico del MCP3204
def read_mcp3204(channel):
    if channel < 0 or channel > 3:
        raise ValueError("El canal debe estar entre 0 y 3")
    
    # Comando para el MCP3204: Start bit + Single/Diff + D2, D1, D0
    cmd = [1, (8 + channel) << 4, 0]
    adc_response = spi.xfer2(cmd)  # Enviar el comando y recibir la respuesta
    result = ((adc_response[1] & 3) << 8) + adc_response[2]  # Procesar la respuesta a un valor de 12 bits
    return result

# Variables para graficar
xdata, ydata = [], []
timelaps = 600  # Número de muestras para mostrar en la gráfica

# Inicializar la figura
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', label="ECG Canal 4")

# Configurar la gráfica
ax.set_xlim(0, timelaps)
ax.set_ylim(0, 4095)  # Rango para 12 bits
ax.set_title("Señal ECG")
ax.set_xlabel("Muestras")
ax.set_ylabel("Valor")
ax.legend()

# Función de inicialización
def init():
    line.set_data([], [])
    return line,

# Función para actualizar la gráfica
def update(frame):
    # Leer el canal 4 (índice 3)
    value = read_mcp3204(3)
    xdata.append(frame)
    ydata.append(value)

    # Mantener el tamaño de los datos en el rango de `timelaps`
    if len(xdata) > timelaps:
        xdata.pop(0)
        ydata.pop(0)

    line.set_data(xdata, ydata)
    return line,

# Animar la gráfica en tiempo real
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, timelaps), init_func=init, blit=True, interval=50)

# Mostrar la gráfica
plt.tight_layout()
plt.show()

# Cerrar el SPI al finalizar
spi.close()
