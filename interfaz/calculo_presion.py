import numpy as np
import matplotlib.pyplot as plt
import requests

def calculo_presion(data1,data2,id_usuario):


    def voltaje(x):
        y=x*3.3/1024
        pre=y*170/3
        return pre


    # Cargar los datos desde los archivos CSV
    #data1 = np.loadtxt("grafica5.csv", delimiter=",", skiprows=1)[:1000]  # Saltar la cabecera
    #data2 = np.loadtxt("grafica6.csv", delimiter=",", skiprows=1)[:1000]

    # Encontrar el valor máximo y su posición en cada conjunto de datos
    max_data1 = np.max(data1)
    max_data1_index = np.argmax(data1)

    max_data2 = np.max(data2)
    max_data2_index = np.argmax(data2)

    # Último punto de datos en la serie de presión de aire (data2)
    last_data2_index = len(data2) - 1  # Índice del último punto
    last_data2_value = data2[-1]       # Valor del último punto

    distance_in_samples = last_data2_index - max_data2_index
    amplitude_difference = last_data2_value - max_data2

    print("Punto de presión oscilometica Mx:", max_data1_index)
    print("Punto de presión de aire Mx:", max_data2_index)
    print("Distancia entre el punto maximo y el ultim dato:", distance_in_samples)
    print()

    distancia_sistole = max_data1_index - max_data2_index
    distancia_diastole = last_data2_index  - max_data1_index
    porciento_sistole = int(np.floor(max_data1_index - (distancia_sistole * 0.6)))
    porciento_diastole = int(np.floor(max_data1_index + (distancia_diastole * 0.4)))
    SISTOLE = data2[porciento_sistole]
    DIASTOLE =data2[porciento_diastole]

    print("Distancia sitole:",porciento_diastole)
    print("Distancia diastole:", porciento_sistole)
    print("Distancia sitole:", distancia_sistole)
    print("Distancia diastole:", distancia_diastole)

    presion_sistolica_final=voltaje(SISTOLE)
    presion_diastolica_final=voltaje(DIASTOLE)

    print("Sistole:", presion_sistolica_final)
    print("Diastole:", presion_diastolica_final)

    url = "http://10.87.30.203:8000/data_user"

    # Datos generados
    data = {"id_usuario":id_usuario,"presion_sistolica": presion_sistolica_final, "presion_diastolica": presion_diastolica_final}

    # Enviar solicitud POST
    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("Datos enviados correctamente:", response.json())
    else:
        print("Error en el envío:", response.status_code, response.text)



    # Crear una figura para las gráficas
    plt.figure(figsize=(8, 6))

    # Gráfica para los datos del primer archivo
    plt.subplot(2, 1, 1)
    plt.plot(data1, label="Canal 1")
    plt.plot(max_data1_index, max_data1, 'ro', label=f'Máximo: {max_data1}')  # Marcar el punto máximo en rojo
    plt.title("Presión Oscilométrica")
    plt.xlabel("Muestras")
    plt.ylabel("Valor")
    plt.legend()

    # Gráfica para los datos del segundo archivo
    plt.subplot(2, 1, 2)
    plt.plot(data2, label="Canal 2", color="orange")
    plt.plot(max_data2_index, max_data2, 'bo', label=f'Máximo: {max_data2}')  # Marcar el punto máximo en azul
    plt.title("Presión de Aire")
    plt.xlabel("Muestras")
    plt.ylabel("Valor")
    plt.legend()

    # Mostrar las gráficas
    plt.tight_layout()
    plt.show()
