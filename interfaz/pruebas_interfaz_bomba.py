import tkinter as tk
from tkinter import ttk
import serial
import time
import spidev
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import numpy as np

# Configura el puerto serial para la comunicación con el Arduino en el puerto COM3
arduino = serial.Serial('/dev/serial0', 9600, timeout=1)
time.sleep(2)  # Espera a que se establezca la conexión serial

# Configurar SPI para MCP3204
spi = spidev.SpiDev()
spi.open(1, 0)
spi.max_speed_hz = 1350000

# Listas para almacenar las lecturas de los sensores
Sensor1 = []
Sensor2 = []
# Variable para controlar la animación
ani = None
timelaps = 2000  # Límite en x
def activar_bomba():
    """Envía el comando para activar la bomba y comienza la animación"""
    global ani
    Sensor1.clear()  # Limpiar listas al activar la bomba
    Sensor2.clear()

    if arduino.isOpen():
        print("Activando bomba...")
        arduino.write(b'1')
        time.sleep(0.1)

        # Inicia la animación al presionar el botón
        ani = animation.FuncAnimation(fig, update, frames=range(timelaps), init_func=init, blit=True, interval=50)
        canvas.draw()

        # Inicia la verificación del estado en un hilo separado
        threading.Thread(target=verificar_estado).start()

def verificar_estado():
    """Verifica el estado de la bomba hasta que se apague"""
    while True:
        arduino.write(b's')
        estado = arduino.readline().decode('utf-8').strip()
        if estado == "0":
            print("Bomba desactivada.")
            break
        else:
            print("Bomba en proceso...")
        time.sleep(0.1)

def funcion_extra():
    """Envía el comando para la función extra"""
    if arduino.isOpen():
        print("Desactivando Bomba")
        arduino.write(b'2')
        time.sleep(0.1)

def cerrar():
    """Cierra la conexión serial y la interfaz"""
    arduino.close()
    root.destroy()
    print("Conexión serial cerrada y aplicación terminada.")

# Configurar la interfaz gráfica principal
root = tk.Tk()
root.attributes("-fullscreen", True)
root.configure(bg="black")

# Crear un frame para los botones (1/3 de la pantalla)
frame_botones = tk.Frame(root, width=root.winfo_screenwidth() // 3, bg="black")
frame_botones.pack(side=tk.LEFT, fill=tk.BOTH)

# Configurar el tamaño de fuente y estilo de los botones
button_font = ("Arial", 32, "bold")

# Botón para activar la bomba
boton_activar = tk.Button(frame_botones, text="Activar Bomba", font=button_font, command=activar_bomba, bg="green", fg="white")
boton_activar.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Botón para la función extra
boton_funcion_extra = tk.Button(frame_botones, text="Desactivador de emergencia", font=button_font, command=funcion_extra, bg="blue", fg="white")
boton_funcion_extra.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Botón para cerrar la aplicación
boton_salir = tk.Button(frame_botones, text="Salir", font=button_font, command=cerrar, bg="red", fg="white")
boton_salir.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Crear una figura de Matplotlib y dos subplots para las gráficas (2/3 de la pantalla)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Inicializar datos para los canales
xdata, ydata_ch1, ydata_ch2 = [], [], []
ln1, = ax1.plot([], [], 'r-', animated=True, label="CH1")
ln2, = ax2.plot([], [], 'b-', animated=True, label="CH2")

# Configurar límites de los gráficos
for ax in (ax1, ax2):
    ax.set_xlim(0, timelaps)
    ax.set_ylim(0, 1027)

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

def update(frame):
    # Leer los datos de los dos canales
    lectura_ch1 = analogRead(0)
    lectura_ch2 = analogRead(1)

    # Almacenar datos en listas y mantener el tamaño de `timelaps`
    if len(Sensor1) < timelaps:
        Sensor1.append(lectura_ch1)
        Sensor2.append(lectura_ch2)
    else:
        # Detener la animación al llenar las listas
        ani.event_source.stop()
        print("Límite alcanzado, guardando datos...")

        # Guardar los datos en archivos CSV
        np.savetxt("grafica5.csv", Sensor1, delimiter=",", header="Valor", comments="")
        np.savetxt("grafica6.csv", Sensor2, delimiter=",", header="Valor", comments="")
        return ln1, ln2  # No actualizar más después de guardar

    # Agregar datos a los arreglos correspondientes
    xdata.append(frame)
    ydata_ch1.append(lectura_ch1)
    ydata_ch2.append(lectura_ch2)

    # Actualizar los datos en las líneas
    ln1.set_data(xdata, ydata_ch1)
    ln2.set_data(xdata, ydata_ch2)

    return ln1, ln2

# Crear un canvas de Matplotlib dentro de tkinter
frame_graficas = tk.Frame(root, width=(root.winfo_screenwidth() * 2) // 3, bg="white")
frame_graficas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
canvas = FigureCanvasTkAgg(fig, master=frame_graficas)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Iniciar el bucle de la interfaz gráfica
plt.tight_layout()
root.mainloop()

# Al finalizar, cierra el SPI
spi.close()