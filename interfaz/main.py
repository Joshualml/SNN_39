import requests
import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
from activacion_sistema import activacion_sistema


def open_keyboard(entry_widget):
    """Abre el teclado virtual y actualiza el texto en la parte superior."""
    global keyboard_process
    top_label.config(text="vacio",bg="#09f",highlightthickness=1)

    # Actualizar el texto dinámicamente en el label superior
    def update_text(*args):
        top_label.config(text=entry_widget.get(),bg="#09f")

    # Conectar el evento de escritura al Entry para actualizar el texto
    entry_widget.bind("<KeyRelease>", lambda event: update_text())
    
    # Mostrar el texto inicial del Entry en el label superior
    top_label.config(text=entry_widget.get())

    # Inicia el teclado virtual Onboard si no está activo
    if keyboard_process is None:
        keyboard_process = subprocess.Popen(["onboard"])


def close_keyboard():
    """Cierra el teclado virtual."""
    global keyboard_process
    if keyboard_process is not None:
        keyboard_process.terminate()
        keyboard_process = None
    # Borra el texto del Label superior
    top_label.config(text="",bg="#30a99b")


def confirmar():
    """Valida las credenciales con el backend."""
    nombre = nombre_entry.get()
    password = password_entry.get()

    data = {"nombre": nombre, "contraseña": password}

    url = "http://10.87.30.203:8000/login"

    # Enviar solicitud POST
    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("Datos enviados correctamente:", response.json())
        data = response.json()
        print(data['id'])
        messagebox.showinfo("Confirmación", f"¡Inicio de sesión exitoso!")
        id_usuario = data['id']
        print(id_usuario)
        
        activacion_sistema(root,id_usuario)  # Cambia la interfaz para mostrar "HOLA"
    else:
        print("Error en el envío:", response.status_code, response.text)
        messagebox.showerror("Error", "Credenciales incorrectas")
    


def scan_networks():
    """Escanea redes Wi-Fi y llena el ComboBox con los SSID."""
    result = subprocess.run(["nmcli", "-t", "-f", "SSID", "dev", "wifi"], capture_output=True, text=True)
    networks = [line for line in result.stdout.strip().splitlines() if line]
    network_combo['values'] = networks
    if networks:
        network_combo.current(0)


def connect_to_wifi():
    """Conecta a la red seleccionada con la contraseña dada."""
    ssid = network_combo.get()
    password = wifi_password_entry.get()
    if not ssid:
        messagebox.showwarning("Red no seleccionada", "Por favor, selecciona una red Wi-Fi.")
        return
    if not password:
        messagebox.showwarning("Contraseña faltante", "Por favor, ingresa la contraseña de la red Wi-Fi.")
        return

    result = subprocess.run(["nmcli", "dev", "wifi", "connect", ssid, "password", password],
                            capture_output=True, text=True)
    if result.returncode == 0:
        messagebox.showinfo("Conexión exitosa", f"Conectado a la red '{ssid}'")
    else:
        messagebox.showerror("Error de conexión", "No se pudo conectar a la red.\n" + result.stderr)


# Crear la ventana principal
root = tk.Tk()
root.geometry("1280x720")
#root.overrideredirect(True)

keyboard_process = None

# Marco principal con dos columnas (izquierda y derecha)
left_frame = tk.Frame(root, width=240, height=320)
left_frame.pack(side="left", fill="both", expand=True)

right_frame = tk.Frame(root, width=240, height=320)
right_frame.pack(side="right", fill="both", expand=True)

# Contenido del marco izquierdo (Inicio de sesión)
login_frame = tk.LabelFrame(left_frame, text="Inicio de Sesión",fg="#a7043a",font=("Arial", 12))
login_frame.config(bg="#30a99b")
login_frame.pack(fill="both", expand=True, padx=0, pady=0)

# Label superior para mostrar el texto actual del Entry
top_label = tk.Entry(login_frame, text="", font=("Arial", 18),bd=0,highlightthickness=0,fg="black",bg="#30a99b")
top_label.place(relx=0.05,rely=0.02,relwidth=0.9,relheight=0.1)


tk.Label(login_frame, text="Nombre:",font=("Arial", 15),fg="#a7043a",bg='#30a99b').place(relx=0.26,rely=0.2)
nombre_entry = tk.Entry(login_frame)
nombre_entry.place(relx=0.075,rely=0.3,relwidth=0.85,relheight=0.1)
nombre_entry.bind("<FocusIn>", lambda event: open_keyboard(nombre_entry))
nombre_entry.bind("<FocusOut>", lambda event: close_keyboard())

tk.Label(login_frame, text="Contraseña:",font=("Arial", 15),fg="#a7043a",bg="#30a99b").place(relx=0.18,rely=0.4)
password_entry = tk.Entry(login_frame, show="*")
password_entry.place(relx=0.075,rely=0.5,relwidth=0.85,relheight=0.1)
password_entry.bind("<FocusIn>", lambda event: open_keyboard(password_entry))
password_entry.bind("<FocusOut>", lambda event: close_keyboard())

btnInicio = tk.Button(login_frame, text="Confirmar",fg="#a7043a",command=confirmar)
btnInicio.config(bg="green")
btnInicio.place(relx=0.3,rely=0.7,relwidth=0.4,relheight=0.08)

# Contenido del marco derecho (Conexión Wi-Fi)
wifi_frame = tk.LabelFrame(right_frame, text="Conexión Wi-Fi",fg="#a7043a",font=("Arial", 12))
wifi_frame.config(bg="#30a99b")
wifi_frame.pack(fill="both", expand=True, padx=0, pady=0)


tk.Label(wifi_frame, text="Selecciona Red Wi-Fi:",font=("Arial", 10)).place(relx=0.05,rely=0.1,relwidth=0.9,relheight=0.10)
network_combo = ttk.Combobox(wifi_frame, state="readonly")
network_combo.place(relx=0.1,rely=0.2,relwidth=0.8,relheight=0.15)
scan_networks()

tk.Label(wifi_frame, text="Contraseña de Wi-Fi:").place(relx=0.1,rely=0.46,relwidth=0.8,relheight=0.15)
wifi_password_entry = tk.Entry(wifi_frame, show="*")
wifi_password_entry.place(relx=0.1,rely=0.63,relwidth=0.8,relheight=0.15)

btnConnect = tk.Button(wifi_frame, text="Conectar", bg="green", command=connect_to_wifi)
btnConnect.place(relx=0.3,rely=0.8,relwidth=0.5,relheight=0.15)

# Cerrar el teclado al cerrar la ventana principal
root.protocol("WM_DELETE_WINDOW", close_keyboard)

# Iniciar el bucle principal
root.mainloop()
