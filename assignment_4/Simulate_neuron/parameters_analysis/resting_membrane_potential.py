import numpy as np
import matplotlib.pyplot as plt

# 1. Load your actual data file
file_path = '/Users/ak/Desktop/Important_folder/NeuroData/Learning_materials/Python_neuroscience/Python_projects/assignment_2a/spike_data.npz'
data = np.load(file_path)

time = data['t']
voltage_trace = data['raw_data']

# Estimate sampling rate (fs) from the time array
dt = time[1] - time[0]
fs = int(np.round(1.0 / dt))

# 2. Detect spikes using a threshold and refractory period
threshold = 4.0  # mV (adjust if your spike peak/onset is lower/higher)
refractory_period_ms = 5.0  # ms to prevent double-counting the same spike
refractory_samples = int((refractory_period_ms / 1000.0) * fs)

# Find upward crossings past the threshold
above_threshold = voltage_trace > threshold
spike_crossings = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1

# Filter by refractory period to isolate distinct spikes
spike_indices = []
last_spike = -refractory_samples
for idx in spike_crossings:
    if idx - last_spike >= refractory_samples:
        spike_indices.append(idx)
        last_spike = idx
spike_indices = np.array(spike_indices)

print(f"Detected {len(spike_indices)} spike(s) in the recording.")

# 3. Calculate V_rest for each spike using a pre-spike window
# (e.g., looking at the window 15ms to 5ms before each spike)
pre_window_start_ms = 15.0
pre_window_end_ms = 5.0

start_offset = int((pre_window_start_ms / 1000.0) * fs)
end_offset = int((pre_window_end_ms / 1000.0) * fs)

v_rests = []
for i, idx in enumerate(spike_indices):
    start_idx = max(0, idx - start_offset)
    end_idx = max(0, idx - end_offset)

    if start_idx < end_idx:
        baseline_segment = voltage_trace[start_idx:end_idx]
        v_rest_val = np.mean(baseline_segment)
        v_rests.append(v_rest_val)

# 4. Print the list of resting membrane potentials
print("\n--- Resting Membrane Potentials (V_rest) Preceding Each Spike ---")
for i, vr in enumerate(v_rests):
    print(f"Spike {i + 1}: {vr:.2f} mV")

if len(v_rests) > 0:
    print(f"\nAverage V_rest across all spikes: {np.mean(v_rests):.2f} mV")