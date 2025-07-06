import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from scipy.interpolate import interp1d

# Filtro pasa-altas para eliminar componentes de baja frecuencia
def highpass_filter(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# Filtro pasa-bajas suavizador
def lowpass_filter(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Cargar datos de la señal original
data1 = np.loadtxt("grafica5.csv", delimiter=",", skiprows=1)  # Cargar la señal desde un archivo CSV

# Parámetros de filtrado
fs = 1000  # Frecuencia de muestreo en Hz, ajústala según tu señal
high_cutoff = 0.1  # Frecuencia de corte para el filtro pasa-altas en Hz
low_cutoff = 5  # Frecuencia de corte para el filtro suavizador en Hz

# Aplicar filtro pasa-altas
filtered_signal = highpass_filter(data1, high_cutoff, fs)

# Método 1: Envolvente usando Transformada de Hilbert + Suavizado
analytic_signal = hilbert(filtered_signal)
hilbert_envelope = np.abs(analytic_signal)
smoothed_envelope = lowpass_filter(hilbert_envelope, low_cutoff, fs)

# Método 2: Envolvente usando Interpolación de Picos con `prominence` y `distance`
# Ajusta `prominence` y `distance` para mejorar la detección de picos.
peaks, _ = find_peaks(filtered_signal, prominence=0.5, distance=10)
peak_values = filtered_signal[peaks]
t = np.arange(len(filtered_signal))
f_interp = interp1d(t[peaks], peak_values, kind='cubic', fill_value="extrapolate")
interpolated_envelope = f_interp(t)

# Graficar los resultados para comparación
plt.figure(figsize=(12, 10))

# Señal Original y Filtrada
plt.subplot(4, 1, 1)
plt.plot(t, data1, label="Señal original", alpha=0.5)
plt.plot(t, filtered_signal, label="Señal filtrada (Pasa-Altas)", color="purple")
plt.title("Señal Original y Señal Filtrada")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()


# Envolvente usando Transformada de Hilbert + Suavizado
plt.subplot(4, 1, 2)
plt.plot(t, filtered_signal, label="Señal Filtrada", color="purple")
plt.plot(t, smoothed_envelope, label="Envolvente Suavizada (Hilbert)", color="red")
plt.title("Envolvente usando Transformada de Hilbert + Filtro Pasa-Bajas Suavizador")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()

# Envolvente usando Interpolación de Picos
plt.subplot(4, 1, 3)
plt.plot(t, filtered_signal, label="Señal Filtrada", color="purple")
plt.plot(t, interpolated_envelope, label="Envolvente (Interpolación de Picos)", color="orange")
plt.title("Envolvente usando Interpolación de Picos con Ajustes")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()

# Zoom en una región para ver detalles
plt.subplot(4, 1, 4)
zoom_region = slice(0, 200)  # Ajusta la región según necesites
plt.plot(t[zoom_region], filtered_signal[zoom_region], label="Señal Filtrada", color="purple")
plt.plot(t[zoom_region], smoothed_envelope[zoom_region], label="Envolvente Suavizada (Hilbert)", color="red")
plt.plot(t[zoom_region], interpolated_envelope[zoom_region], label="Envolvente (Interpolación de Picos)", color="orange")
plt.title("Zoom en la Señal Filtrada y Envolventes")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()

plt.tight_layout()
plt.show()