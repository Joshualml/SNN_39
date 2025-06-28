import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# === Modelo CNN-1D Profunda ===
class DeepCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),  # 1700 → 850
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), # 850 → 425
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2)  # 425 → 212
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 212, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# === Cargar modelo y escaladores ===
model = DeepCNN1D()
model.load_state_dict(torch.load("modelo_cnn1d_profundo.pt"))
model.eval()

scaler_X = joblib.load("scaler_X.gz")
scaler_y = joblib.load("scaler_y.gz")

# === Función para usar desde otro script ===
def estimar_presion_cnn_profunda(senal_osc):
    senal_osc = np.array(senal_osc).flatten()
    if senal_osc.shape[0] != 1700:
        raise ValueError("La señal debe tener exactamente 1700 muestras")

    X_scaled = scaler_X.transform(senal_osc.reshape(1, -1))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)  # (1, 1, 1700)

    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()
        pred_real = scaler_y.inverse_transform(pred_scaled)

    return pred_real.flatten()

# === Ejecución directa desde archivo CSV ===
if __name__ == "__main__":
    try:
        senal = pd.read_csv("senal_temporal.csv", header=None).values.flatten()
        resultado = estimar_presion_cnn_profunda(senal)
        print(f"✅ CNN-1D Profunda: Sistólica = {resultado[0]:.1f} mmHg | Diastólica = {resultado[1]:.1f} mmHg")
    except Exception as e:
        print(f"❌ Error en inferencia CNN-1D Profunda: {e}")
