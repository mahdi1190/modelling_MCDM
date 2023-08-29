import numpy as np
import matplotlib.pyplot as plt

# Time array
t = np.linspace(0, 10, 1000)
tau = 5  # space time

# Plug Flow
pf_pulse = np.where(t == tau, 1, 0)
pf_step = np.where(t >= tau, 1, 0)

# CSTR
cstr_pulse = np.exp(-t/tau)
cstr_step = 1 - np.exp(-t/tau)

# Bypass (assuming 30% bypasses immediately)
bypass_fraction = 0.3
bypass_pulse = bypass_fraction * np.where(t == 0, 1, 0) + (1 - bypass_fraction) * np.where(t == tau, 1, 0)
bypass_step = bypass_fraction + (1 - bypass_fraction) * np.where(t >= tau, 1, 0)

# Dead Volume (assuming a 30% delay in effective volume)
dead_time = 1.3 * tau
dead_pulse = np.where(t == dead_time, 1, 0)
dead_step = np.where(t >= dead_time, 1, 0)

# Recirculation (simplistic representation with sine wave, might need more realistic modeling for specific cases)
recirculation_pulse = np.sin(t/tau * 2 * np.pi) * np.exp(-t/tau)
recirculation_step = 0.5 + 0.5 * np.sin(t/tau * 2 * np.pi) * np.exp(-t/tau)

# Visualize
plt.figure(figsize=(15, 10))

# Plug Flow
plt.subplot(2, 3, 1)
plt.plot(t, pf_pulse, label="Pulse Input")
plt.plot(t, pf_step, label="Step Input", linestyle='--')
plt.title("Plug Flow")
plt.legend()

# CSTR
plt.subplot(2, 3, 2)
plt.plot(t, cstr_pulse, label="Pulse Input")
plt.plot(t, cstr_step, label="Step Input", linestyle='--')
plt.title("CSTR")
plt.legend()

# Bypass
plt.subplot(2, 3, 3)
plt.plot(t, bypass_pulse, label="Pulse Input")
plt.plot(t, bypass_step, label="Step Input", linestyle='--')
plt.title("Bypass")
plt.legend()

# Dead Volume
plt.subplot(2, 3, 4)
plt.plot(t, dead_pulse, label="Pulse Input")
plt.plot(t, dead_step, label="Step Input", linestyle='--')
plt.title("Dead Volume")
plt.legend()

# Recirculation
plt.subplot(2, 3, 5)
plt.plot(t, recirculation_pulse, label="Pulse Input")
plt.plot(t, recirculation_step, label="Step Input", linestyle='--')
plt.title("Recirculation")
plt.legend()

plt.tight_layout()
plt.show()


# Time array
t = np.linspace(0, 10, 1000)
tau = 5  # space time

# Input Tracer Concentrations
pulse_input = np.where(t < 0.1, 1, 0)  # Small pulse at t=0
step_input = np.where(t > 0, 1, 0)     # Step function starting immediately at t=0

# CSTR Output Response to the Tracer Inputs
cstr_pulse_output = np.convolve(pulse_input, np.exp(-t/tau), mode='same')
cstr_step_output = 1 - np.exp(-t/tau) 

# Visualize
plt.figure(figsize=(12, 6))

# Pulse Input vs. Output
plt.subplot(1, 2, 1)
plt.plot(t, pulse_input, label="Pulse Input", linestyle='--')
plt.plot(t, cstr_pulse_output, label="CSTR Output")
plt.title("Pulse Input & CSTR Response")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()

# Step Input vs. Output
plt.subplot(1, 2, 2)
plt.plot(t, step_input, label="Step Input", linestyle='--')
plt.plot(t, cstr_step_output, label="CSTR Output")
plt.title("Step Input & CSTR Response")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()

plt.tight_layout()
plt.show()