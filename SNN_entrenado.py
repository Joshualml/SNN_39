import torch
import numpy as np
import snntorch as snn
import torch.nn as nn

# === Configuración del modelo ===
n_bins = 50  # Debe coincidir con el valor usado en entrenamiento
beta = 0.9

class DeepSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_bins, 64)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(64, 32)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x, num_steps=20):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            _, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(mem1)
            _, mem2 = self.lif2(cur2, mem2)
        out = self.fc3(mem2)
        return out

# === Función para hacer la predicción ===
def predecir_presion(picos_x_temp, picos_y_temp, modelo_path="best_snn_model6.pt"):
    if len(picos_x_temp) == 0:
        raise ValueError("El arreglo de picos está vacío")

    # Codificación por tasa (rate coding)
    time_array = np.array(picos_x_temp, dtype=float)
    time_norm = (time_array - time_array.min()) / (time_array.max() - time_array.min())
    spike_counts, _ = np.histogram(time_norm, bins=n_bins, range=(0, 1))
    X_input = torch.tensor(spike_counts, dtype=torch.float32).unsqueeze(0)  # [1, 50]

    # Cargar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSNN().to(device)
    model.load_state_dict(torch.load(modelo_path, map_location=device))
    model.eval()

    # Predicción
    with torch.no_grad():
        X_input = X_input.to(device)
        salida = model(X_input)

    return salida.squeeze().cpu().numpy()  # Devuelve [presión sistólica, presión diastólica]