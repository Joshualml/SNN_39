import requests
import tkinter as tk
from tkinter import messagebox
import json
import os
import subprocess

# Crear la ventana principal
root = tk.Tk()
root.title("Formulario de Login")
root.geometry("300x200")

# Variable para almacenar el proceso del teclado virtual
keyboard_process = None

# Función para abrir el teclado virtual
def open_keyboard():
    global keyboard_process
    if keyboard_process is None:  # Evita abrir múltiples instancias
        keyboard_process = subprocess.Popen(["matchbox-keyboard"])

# Función para cerrar el teclado virtual
def close_keyboard():
    global keyboard_process
    if keyboard_process is not None:
        keyboard_process.terminate()
        keyboard_process = None

# Etiqueta y campo de entrada para el nombre
tk.Label(root, text="Nombre:").pack(pady=5)
nombre_entry = tk.Entry(root)
nombre_entry.pack(pady=5)
nombre_entry.bind("<FocusIn>", lambda event: open_keyboard())  # Abre el teclado al seleccionar
nombre_entry.bind("<FocusOut>", lambda event: close_keyboard())  # Cierra el teclado al deseleccionar

# Etiqueta y campo de entrada para la contraseña
tk.Label(root, text="Contraseña:").pack(pady=5)
password_entry = tk.Entry(root, show="*")
password_entry.pack(pady=5)
password_entry.bind("<FocusIn>", lambda event: open_keyboard())  # Abre el teclado al seleccionar
password_entry.bind("<FocusOut>", lambda event: close_keyboard())  # Cierra el teclado al deseleccionar

# Función para manejar el clic en el botón de confirmación
def confirmar():
    nombre = nombre_entry.get()
    password = password_entry.get()


    data = {"nombre": nombre, "contraseña": password}

    url = "http://https://e6b1-2806-2f0-a701-e68a-6450-da17-5845-5fa6.ngrok-free.app/login"

    # Enviar solicitud POST
    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("Datos enviados correctamente:", response.json())
    else:
        print("Error en el envío:", response.status_code, response.text)

    # Mostrar mensaje de confirmación
    messagebox.showinfo("Confirmación", f"Nombre y contraseña guardados")

    # Ejecutar el segundo programa (asegúrate de que 'second_window.py' esté en la misma carpeta)
    root.destroy()  # Cierra la ventana actual
    os.system("python second_window.py")

# Botón de confirmación
confirm_button = tk.Button(root, text="Confirmar", command=confirmar)
confirm_button.pack(pady=20)

# Cerrar el teclado al cerrar la ventana principal
root.protocol("WM_DELETE_WINDOW", close_keyboard)

# Iniciar el bucle principal
root.mainloop()


