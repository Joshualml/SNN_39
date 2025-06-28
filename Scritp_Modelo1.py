import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

'''

from Script_Modelo1 import estimar_presion

# Ejemplo de señal generada desde sensor (np.array de tamaño 1700)
senal = obtener_senal_arduino()  # deberías obtener un array de tamaño 1700

presion = estimar_presion(senal)
print("Presión estimada:", presion)

'''

# === Modelo CNN-1D (igual al usado en entrenamiento) ===
class CNN1DRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32 * 425, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)

# === Cargar modelo y escaladores ===
model = CNN1DRegressor()
model.load_state_dict(torch.load("modelo_cnn1d.pt"))
model.eval()

scaler_X = joblib.load("scaler_X.gz")
scaler_y = joblib.load("scaler_y.gz")

# === Función principal para usar desde otro script ===
def estimar_presion(senal_osc):
    """
    senal_osc: numpy array de forma (1700,) o (1, 1700)
    Devuelve: array [sistolica, diastolica] en mmHg
    """
    senal_osc = np.array(senal_osc).flatten()
    
    if senal_osc.shape[0] != 1700:
        raise ValueError("La señal debe tener exactamente 1700 muestras")

    # Escalar y convertir a tensor
    X_scaled = scaler_X.transform(senal_osc.reshape(1, -1))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)

    # Inferencia
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()
        pred_real = scaler_y.inverse_transform(pred_scaled)
    
    return pred_real.flatten()

# === Ejemplo: cargar señal desde CSV y predecir ===
if __name__ == "__main__":
    try:
        senal_csv = pd.read_csv("senal_temporal.csv", header=None).values.flatten()
        resultado = estimar_presion(senal_csv)
        print(f"✅ Presión estimada: Sistólica = {resultado[0]:.1f} mmHg | Diastólica = {resultado[1]:.1f} mmHg")
    except Exception as e:
        print(f"❌ Error en la inferencia: {e}")
