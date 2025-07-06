import spidev
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configurar SPI
spi = spidev.SpiDev()
spi.open(1, 2)  # Ajusta según tu configuración de SPI (por ejemplo, SPI0, CE0)
spi.max_speed_hz = 1350000  # Establecer velocidad adecuada para el MCP3204

# Inicializar la figura y dos subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # 2 filas, 1 columna

# Inicializar datos para los canales
xdata, ydata_ch1, ydata_ch2 = [], [], []
ln1, = ax1.plot([], [], 'r-', animated=True, label="CH1")
ln2, = ax2.plot([], [], 'b-', animated=True, label="CH2")

# Configurar límites de los gráficos
timelaps = 700
for ax in (ax1, ax2):
    ax.set_xlim(0, timelaps)  # Rango de tiempo
    ax.set_ylim(0, 1024)  # Rango de 12 bits para el MCP3204 (0-4095)

ax1.set_title("Señal del Canal 1")
ax2.set_title("Señal del Canal 2")

# Función para leer del ADC
def analogRead(pin):
    adc = spi.xfer2([1, (8 + pin) << 4, 0])
    lec = ((adc[1] & 3) << 8) + adc[2]
    return lec

# Función para inicializar la gráfica
def init():
    ln1.set_data([], [])
    ln2.set_data([], [])
    return ln1, ln2

# Función para actualizar los datos de las gráficas
def update(frame):
    # Leer los datos de los dos canales
    lectura_ch1 = analogRead(0)  # Leer del canal 0
    lectura_ch2 = analogRead(1)  # Leer del canal 1

    # Agregar datos a los arreglos correspondientes
    xdata.append(frame)
    ydata_ch1.append(lectura_ch1)
    ydata_ch2.append(lectura_ch2)

    # Limitar la cantidad de puntos en pantalla
    if len(xdata) > timelaps:
        xdata.pop(0)
        ydata_ch1.pop(0)
        ydata_ch2.pop(0)

    # Actualizar los datos en las líneas
    ln1.set_data(xdata, ydata_ch1)
    ln2.set_data(xdata, ydata_ch2)
    return ln1, ln2

# Animar los datos de las gráficas en tiempo real
ani = animation.FuncAnimation(fig, update, frames=range(timelaps),
                              init_func=init, blit=True, interval=50)

# Mostrar la gráfica
plt.tight_layout()
plt.show()

# Al finalizar, cierra el SPI
spi.close()