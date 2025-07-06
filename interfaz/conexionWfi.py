import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

class WiFiConnectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Conectar a Wi-Fi")

        # Etiqueta de redes
        tk.Label(root, text="Selecciona Red Wi-Fi:").pack(pady=5)

        # ComboBox para seleccionar la red Wi-Fi
        self.network_combo = ttk.Combobox(root, state="readonly")
        self.network_combo.pack(pady=5)
        self.scan_networks()

        # Etiqueta y entrada de contraseña
        tk.Label(root, text="Contraseña:").pack(pady=5)
        self.password_entry = tk.Entry(root, show="*")
        self.password_entry.pack(pady=5)

        # Botón de conectar
        connect_button = tk.Button(root, text="Conectar", command=self.connect_to_wifi)
        connect_button.pack(pady=10)

    def scan_networks(self):
        """Escanea redes Wi-Fi y llena el ComboBox con los SSID."""
        result = subprocess.run(["nmcli", "-t", "-f", "SSID", "dev", "wifi"], capture_output=True, text=True)
        networks = [line for line in result.stdout.strip().splitlines() if line]
        self.network_combo['values'] = networks
        if networks:
            self.network_combo.current(0)

    def connect_to_wifi(self):
        """Conecta a la red seleccionada con la contraseña dada."""
        ssid = self.network_combo.get()
        password = self.password_entry.get()
        if not ssid:
            messagebox.showwarning("Red no seleccionada", "Por favor, selecciona una red Wi-Fi.")
            return
        if not password:
            messagebox.showwarning("Contraseña faltante", "Por favor, ingresa la contraseña de la red Wi-Fi.")
            return

        # Intenta conectarse a la red Wi-Fi
        result = subprocess.run(["nmcli", "dev", "wifi", "connect", ssid, "password", password],
                                capture_output=True, text=True)
        if result.returncode == 0:
            messagebox.showinfo("Conexión exitosa", f"Conectado a la red '{ssid}'")
        else:
            messagebox.showerror("Error de conexión", "No se pudo conectar a la red.\n" + result.stderr)

# Configuración de la ventana principal
root = tk.Tk()
app = WiFiConnectApp(root)
root.mainloop()