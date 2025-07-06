import tkinter as tk
import json

# Leer los datos del archivo JSON
with open("datos.json", "r") as f:
    data = json.load(f)

# Crear una nueva ventana
root = tk.Tk()
root.title("Segunda Ventana")
root.geometry("300x150")

# Mostrar el nombre y la contraseña guardados
tk.Label(root, text=f"Nombre: {data['nombre']}").pack(pady=10)
tk.Label(root, text=f"Contraseña: {data['password']}").pack(pady=10)

# Iniciar el bucle principal
root.mainloop()