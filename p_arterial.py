import spidev
import csv
import os
import serial
from collections import deque
from time import sleep
from SNN_entrenado_2 import predecir_presion

# Configurar SPI
spi = spidev.SpiDev()
spi.open(1, 0)
spi.max_speed_hz = 1350000


# Leer canal MCP3008
def leer_canal(channel=1):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    return ((adc[1] & 3) << 8) + adc[2]

# Puerto serial con Arduino
arduino = serial.Serial('/dev/serial0', 9600, timeout=1)
sleep(2)

# Enviar comando inicial al Arduino
arduino.write(b'2')
print("Comando '2' enviado al Arduino, comenzando adquisición...")

# Pre-inflado
print("Pre-inflado")
for _ in range(150):
    leer_canal(1)
    sleep(0.01)

# Parámetros
ventana_filtro = 10
buffer_filtro = deque([0]*ventana_filtro, maxlen=ventana_filtro)
ultimos_valores_filtrados = deque(maxlen=3)
timelaps = 1700
y_data = []


picos_x = []
picos_y = []
pico_maximo = 0
inicio_conteo = 0
picos_descendentes = 0
accion_enviada = False
estado_pendiente_descendiendo = False


# Filtro
def filtro_media_movil(nuevo_valor):
    buffer_filtro.append(nuevo_valor)
    return sum(buffer_filtro) / len(buffer_filtro)

# Detección de picos
def detectar_pico(valores, umbral=0):
    global estado_pendiente_descendiendo
    if len(valores) < 3:
        return False
    y1, y2, y3 = valores[-3], valores[-2], valores[-1]
    if (y3 - y2) < 0 and not estado_pendiente_descendiendo:
        estado_pendiente_descendiendo = True
        return y2 > umbral
    elif (y3 - y2) >= 0:
        estado_pendiente_descendiendo = False
    return False


# Adquisición
for contador in range(timelaps):
    valor = leer_canal(1)
    valor_ajustado = -1 * valor + 1027
    valor_filtrado = filtro_media_movil(valor_ajustado)
    y_data.append(valor_filtrado)
    ultimos_valores_filtrados.append(valor_filtrado)

    if detectar_pico(ultimos_valores_filtrados, umbral=10):
        x_pico = contador
        y_pico = ultimos_valores_filtrados[-2]
        picos_x.append(x_pico)
        picos_y.append(y_pico)

        if y_pico > pico_maximo:
            pico_maximo = y_pico
            inicio_conteo += 1
            picos_descendentes = 0
        else:
            inicio_conteo += 1
            picos_descendentes += 1
            if inicio_conteo > 10 and picos_descendentes >= 5 and not accion_enviada:
                arduino.write(b'1')
                sleep(0.75)
                arduino.write(b'1')
                accion_enviada = True
                print("5 picos descendentes detectados. Señal '1' enviada al Arduino.")

    sleep(0.02)

print("Se alcanzaron 1700 datos.")

# Guardar
respuesta = input("¿Deseas guardar esta señal? (s/n): ").strip().lower()
if respuesta == 's':

    def append_fila_csv(nombre_archivo, nueva_fila):
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

    append_fila_csv('senal_v2.csv', y_data)
    append_fila_csv('picos_v2.csv', picos_x)
    append_fila_csv('picos_v2.csv', picos_y)

    presiones_finales = predecir_presion(picos_x, picos_y)
    print(f'Presión sistolica:{presiones_finales[0]:.2f}')
    print(f'Presión sistolica:{presiones_finales[1]:.2f}')

    print("Señal guardada.")
else:
    print("Señal descartada.")
