import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt

# 1. Load your actual data file
file_path = '/Users/ak/Desktop/Important_folder/NeuroData/Learning_materials/Python_neuroscience/Python_projects/assignment_2a/spike_data.npz'
data = np.load(file_path)

time = data['t']
raw_data = data['raw_data']

# Estimate sampling rate (fs) from time array
dt = time[1] - time[0]
fs = int(np.round(1.0 / dt))

# 2. Step 1: Remove/suppress spikes using a median filter
window_samples = int(0.01 * fs)  # 10 ms window
if window_samples % 2 == 0:
    window_samples += 1  # must be odd for medfilt
spike_removed_trace = medfilt(raw_data, kernel_size=window_samples)

# 3. Step 2: Apply a Low-Pass Butterworth Filter to isolate the drift
def lowpass_filter(data, cutoff_hz, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

estimated_drift = lowpass_filter(spike_removed_trace, cutoff_hz=0.5, fs=fs)

# 4. Step 3: Create a synthetic sinusoidal drift curve matching ~0.0908 Hz
f_drift = 0.0908  # Hz
amplitude = (np.max(estimated_drift) - np.min(estimated_drift)) / 2
mean_offset = np.mean(estimated_drift)

# Generate pure sinusoidal wave with matched frequency, amplitude, and offset
# (You can adjust phase shift inside the sin argument if needed, e.g., + phase)
synthetic_sine = mean_offset + amplitude * np.sin(2 * np.pi * f_drift * time)

# 5. Plot the raw data, filtered drift estimate, and synthetic sine wave together
plt.figure(figsize=(12, 5))
plt.plot(time, raw_data, color='lightgray', alpha=0.6, label='Raw Data (with spikes & noise)')
plt.plot(time, estimated_drift, color='crimson', lw=2, label='Filtered Baseline Drift (Low-Pass)')
plt.plot(time, synthetic_sine, color='dodgerblue', linestyle='--', lw=1.5, label=f'Synthetic Sine Curve ({f_drift} Hz)')

plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (a.u. / mV)')
plt.title('Neural Recording with Extracted Drift and Overlapped Sinusoidal Model')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()