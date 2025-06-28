import torch
import numpy as np
import snntorch as snn
import torch.nn as nn

# === Codificación binaria por presencia de spike ===
def codificacion_binaria(tiempos, longitud=1700):
    vector = np.zeros(longitud)
    for t in tiempos:
        idx = int(round(t))
        if 0 <= idx < longitud:
            vector[idx] = 1
    return vector

# === Definición del modelo (igual al usado en entrenamiento) ===
class DeepSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1700, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(6, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.lif2 = snn.Leaky(beta=0.9)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x, num_steps=20):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        for _ in range(num_steps):
            cur1 = self.dropout1(self.fc1(x))
            _, mem1 = self.lif1(cur1, mem1)
            cur2 = self.dropout2(self.fc2(mem1))
            _, mem2 = self.lif2(cur2, mem2)
        out = self.fc3(mem2)
        return out

# === Función de predicción ===
def predecir_presion(picos_x, picos_y, modelo_path="best_snn_model6.pt"):
    modelo = DeepSNN()
    modelo.load_state_dict(torch.load(modelo_path, map_location=torch.device('cpu')))
    modelo.eval()

    # Usamos solo picos_x para codificación binaria
    entrada = codificacion_binaria(np.array(picos_x))
    entrada_tensor = torch.tensor(entrada, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        salida = modelo(entrada_tensor)

    presion = salida.squeeze().numpy()
    return presion  # [sistólica, diastólica]