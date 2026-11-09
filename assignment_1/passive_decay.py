import numpy as np
import matplotlib.pyplot as plt

# 1. Set up simulation parameters
t_start = 0.0          # Start time (ms)
t_end = 50.0           # End time (ms)
dt = 0.1               # Time step (ms)
time = np.arange(t_start, t_end, dt)

V_rest = -70.0         # Resting membrane potential (mV)
tau = 5.0              # Membrane time constant (ms)
V_peak = -40.0         # Voltage right after the stimulus hits (mV)

# 2. Calculate passive voltage decay
# Formula: V(t) = V_rest + (V_peak - V_rest) * exp(-t / tau)
voltage = V_rest + (V_peak - V_rest) * np.exp(-time / tau)

# 3. Plot the results
plt.figure(figsize=(8, 4))
plt.plot(time, voltage, color='b', linewidth=2, label='Membrane Potential')
plt.axhline(y=V_rest, color='r', linestyle='--', label='Resting Potential (-70 mV)')

plt.title('Passive Voltage Decay of a Single Neuron')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.ylim(-75, -35)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()



