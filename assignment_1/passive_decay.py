import numpy as np
import matplotlib.pyplot as plt

# Parameters
A = 60.0              # Initial peak voltage in mV
half_life = 12.0      # Half-life in ms
duration = 250.0      # Total duration in ms
sampling_freq = 20000 # Sampling frequency in Hz (20,000 samples/sec = 20 samples/ms)

# Time step in milliseconds (dt = 1 / 20 = 0.05 ms)
dt = 1.0 / (sampling_freq / 1000.0)

# Time vector from 0 to 250 ms
t = np.arange(0, duration + dt, dt)

# Calculate decay constant gamma (using milliseconds)
gamma = np.log(2) / half_life

# Compute exponential voltage decay
v = A * np.exp(-gamma * t)

# Plotting the simulation
plt.figure(figsize=(10, 5))
plt.plot(t, v, label='Membrane Voltage $v(t)$', color='b', linewidth=1.5)
plt.axvline(half_life, color='r', linestyle='--', label=f'Half-life ($t_{{1/2}}$ = {half_life} ms)')
plt.axhline(A / 2, color='g', linestyle=':', label=f'Half-voltage ({A/2} mV)')

plt.title('Passive Voltage Decay of a Single Neuron Membrane')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
