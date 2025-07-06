
import requests

url = "http://10.87.4.155:8000/data"

# Datos generados
params = {"data": 8}  # Parámetro de consulta en la URL

# Enviar solicitud GET con parámetros
response = requests.post(url, json=params)

if response.status_code == 200:
    print("Datos enviados correctamente:", response.json())
else:
    print("Error en el envío:", response.status_code, response.text)


