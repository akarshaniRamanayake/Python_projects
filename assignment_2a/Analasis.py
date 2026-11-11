import numpy as np
import matplotlib.pyplot as plt

# Load the .npz archive
file_path = "/Users/ak/Desktop/Important_folder/NeuroData/Learning_materials/Python_neuroscience/Python_projects/assignment_2a/spike_data.npz"
data = np.load(file_path)

# Display the keys (array names) stored inside the file
print("Available keys in spike_data.npz:", data.files)

# Inspect the shape and type of each stored array
for key in data.files:
    arr = data[key]
    print(f"\nKey: '{key}'")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Preview (first 5 elements): {arr.flatten()[:5]}")



# Load the archive
file_path = "/Users/ak/Desktop/Important_folder/NeuroData/Learning_materials/Python_neuroscience/Python_projects/assignment_2a/spike_data.npz"
data = np.load(file_path)

t = data['t']
raw_data = data['raw_data']

# Plot the full neural recording trace
plt.figure(figsize=(12, 4))
plt.plot(t, raw_data, color='black', linewidth=0.8, label='Raw Signal')
plt.title('Neural Recording Trace')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude / Voltage')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()