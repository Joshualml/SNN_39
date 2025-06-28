import spidev
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import csv
import os
import serial
from time import sleep
from SNN_entrenado_2 import predecir_presion



# Configurar SPI
spi = spidev.SpiDev()
spi.open(1, 0)  # SPI1, CE0 (ajusta según tu conexión)
spi.max_speed_hz = 1350000


# Función para leer canal MCP3008 (canal 1)
def leer_canal(channel=1):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

# Configurar puerto serial con Arduino
arduino = serial.Serial('/dev/serial0', 9600, timeout=1)
sleep(2)  # Esperar a que se establezca conexión serial


def append_fila_csv(nombre_archivo, nueva_fila):
    import os
    import csv

    # Crear carpeta si no existe
    carpeta = os.path.dirname(nombre_archivo)
    if carpeta and not os.path.exists(carpeta):
        os.makedirs(carpeta)

    # Leer contenido existente (si existe)
    filas_existentes = []
    if os.path.isfile(nombre_archivo):
        with open(nombre_archivo, 'r', newline='') as f:
            reader = csv.reader(f)
            filas_existentes = list(reader)

    # Agregar la nueva fila
    filas_existentes.append(nueva_fila)

    # Escribir todo de nuevo
    with open(nombre_archivo, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(filas_existentes)

# --- PASO 2: Enviar comando '2' al Arduino ---
arduino.write(b'2')
print("Comando '2' enviado al Arduino, comenzando graficación...")

# --- PASO 1: Leer 1000 valores y calcular promedio ---
valores = []
print("Pre-inflado")
for _ in range(150):
    val = leer_canal(1)
    valores.append(val)
    sleep(0.01)  # Ajusta el tiempo según velocidad de muestreo deseada (10ms = 100Hz aprox)


ventana_filtro = 10
buffer_filtro = deque([0]*ventana_filtro, maxlen=ventana_filtro)
ultimos_valores_filtrados = deque(maxlen=3)


picos_x = []
picos_y = []
promedios_picos_y = []
promedios_picos_x = []


fig, ax = plt.subplots()
timelaps = 1700
ax.set_xlim(0, timelaps)
ax.set_ylim(0, 1023)

x_data = list(range(timelaps))
y_data = [0] * timelaps

line, = ax.plot(x_data, y_data, label="Señal filtrada")
pico_line, = ax.plot([], [], 'ro', label="Picos")
promedio_line, = ax.plot([], [], 'g--', label="Prom. picos")

contador = 0

pico_maximo = 0
inicio_conteo = 0
picos_descendentes = 0
accion_enviada = False

senal_filtrada_temp = []
picos_x_temp = []
picos_y_temp = []

def filtro_media_movil(nuevo_valor):
    buffer_filtro.append(nuevo_valor)
    return sum(buffer_filtro) / len(buffer_filtro)

estado_pendiente_descendiendo = False

def detectar_pico(valores, umbral=0):
    global estado_pendiente_descendiendo

    if len(valores) < 3:
        return False

    y1, y2, y3 = valores[-3], valores[-2], valores[-1]
    pendiente_1 = y2 - y1
    pendiente_2 = y3 - y2

    # Si la pendiente cambia de subida/plana a bajada
    if pendiente_2 < 0 and not estado_pendiente_descendiendo:
        estado_pendiente_descendiendo = True
        if y2 > umbral:
            return True  # Es el primer punto de una bajada ⇒ pico

    # Si ya estamos en bajada, ignoramos
    elif pendiente_2 >= 0:
        estado_pendiente_descendiendo = False  # Reseteamos cuando vuelve a subir

    return False
x=True

while (x):


    sleep(0.02)
    if contador < timelaps:

        valor = leer_canal(1)
        valor_ajustado = -1 * valor + 1027

        valor_filtrado = filtro_media_movil(valor_ajustado)
        ultimos_valores_filtrados.append(valor_filtrado)

        y_data.append(valor_filtrado)
        y_data.pop(0)
        line.set_ydata(y_data)

        if detectar_pico(ultimos_valores_filtrados, umbral=10):
            x_pico = contador
            y_pico = ultimos_valores_filtrados[-2]
            picos_x.append(x_pico)
            picos_y.append(y_pico)

            promedio_actual = sum(picos_y) / len(picos_y)
            promedios_picos_y.append(promedio_actual)
            promedios_picos_x.append(x_pico)

            #print(f"Pico detectado en x={x_pico}, valor={y_pico}")

            if y_pico > pico_maximo:
                pico_maximo = y_pico
                inicio_conteo += 1
                picos_descendentes = 0
                print(f"Nuevo pico máximo actualizado: {pico_maximo}")

            else:
                inicio_conteo += 1
                picos_descendentes += 1

                if inicio_conteo > 10 and y_pico < pico_maximo:
                    print(f"Pico menor detectado: {y_pico})")

                    if (picos_descendentes >= 5 and not accion_enviada):
                        arduino.write(b'1')
                        sleep(0.75)
                        arduino.write(b'1')
                        accion_enviada = True
                        print("Al menos 5 picos menores detectados. Señal '1' enviada al Arduino.")


        # Graficar picos
        # Actualizar líneas
        pico_line.set_data(picos_x, picos_y)
        promedio_line.set_data(promedios_picos_x, promedios_picos_y)

        contador += 1


    else:


        # Guardar temporalmente los datos
        senal_filtrada_temp = list(y_data)
        picos_x_temp = list(picos_x)
        picos_y_temp = list(picos_y)

        # Preguntar en consola si guardar
        respuesta = input("¿Deseas guardar esta señal? (s/n): ").strip().lower()

        if respuesta == 's':
            def append_fila_csv(nombre_archivo, nueva_fila):
                import os, csv
                carpeta = os.path.dirname(nombre_archivo)
                if carpeta and not os.path.exists(carpeta):
                    os.makedirs(carpeta)

                filas_existentes = []
                if os.path.isfile(nombre_archivo):
                    with open(nombre_archivo, 'r', newline='') as f:
                        reader = csv.reader(f)
                        filas_existentes = list(reader)

                filas_existentes.append(nueva_fila)

                with open(nombre_archivo, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(filas_existentes)

            append_fila_csv('senal_v2.csv', senal_filtrada_temp)
            append_fila_csv('picos_v2.csv', picos_x_temp)
            append_fila_csv('picos_v2.csv', picos_y_temp)

            presiones_finales = predecir_presion(picos_x_temp, picos_y_temp)
            print(f'Presión sistolica:{presiones_finales[0]:.2f}')
            print(f'Presión sistolica:{presiones_finales[1]:.2f}')

            print("Señal guardada.")
        else:
            print("Señal descartada.")

        print("Se alcanzaron 17(00 datos.")


        x = False

ax.legend()
plt.show()