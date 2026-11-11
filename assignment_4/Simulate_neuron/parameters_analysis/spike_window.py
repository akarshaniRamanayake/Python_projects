import numpy as np
import matplotlib.pyplot as plt

# 1. Load your actual data file
file_path = '/Users/ak/Desktop/Important_folder/NeuroData/Learning_materials/Python_neuroscience/Python_projects/assignment_2a/spike_data.npz'
data = np.load(file_path)

# Extract arrays using the correct keys found in your archive
time = data['t']
voltage_trace = data['raw_data']

# If your time array is in seconds, we can find a spike or specify a center time.
# Let's inspect where the peak is, or pick a window around the maximum voltage:
peak_idx = np.argmax(voltage_trace)
spike_center_sec = time[peak_idx]

print(f"Peak voltage found at index {peak_idx}, time = {spike_center_sec:.4f} seconds")


# 2. Define plotting function for the real data
def plot_zoomed_spike_real(time_sec, voltage_trace, spike_center_sec, window_ms=20):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    # --- Panel 1: Full Overview Trace ---
    ax1.plot(time_sec, voltage_trace, color='k', lw=0.8)
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title('Full Voltage Trace Overview (Real Data)')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Highlight the zoom region on the overview plot
    half_window_sec = (window_ms / 1000.0) / 2.0
    zoom_start = spike_center_sec - half_window_sec
    zoom_end = spike_center_sec + half_window_sec

    ax1.axvspan(zoom_start, zoom_end, color='crimson', alpha=0.2, label=f'Zoom Window ({window_ms} ms)')
    ax1.legend(loc='upper right')

    # --- Panel 2: Zoomed-In Spike View ---
    ax2.plot(time_sec * 1000, voltage_trace, color='k', lw=1.2, marker='o', ms=2)
    ax2.set_xlim(zoom_start * 1000, zoom_end * 1000)  # Convert to ms for plotting
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.set_title(f'Zoomed-in Action Potential (Around {spike_center_sec * 1000:.1f} ms)')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


# 3. Execute the plot with your real data
plot_zoomed_spike_real(time, voltage_trace, spike_center_sec, window_ms=300)