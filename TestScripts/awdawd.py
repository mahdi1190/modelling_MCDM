import numpy as np
import matplotlib.pyplot as plt

# Time array
t = np.linspace(0, 10, 1000)
tau = 5  # space time

# Step Input Tracer Concentration
step_input = np.where(t > 0, 1, 0)

# Plug Flow Output Response
pf_step_output = np.where(t >= tau, 1, 0)

# CSTR Output Response
cstr_step_output = 1 - np.exp(-t/tau)

# Bypass Output Response (assuming 30% bypass)
bypass_fraction = 0.3
bypass_step_output = bypass_fraction + (1 - bypass_fraction) * np.where(t >= tau, 1, 0)

# Dead Volume Output Response (assuming a 30% delay)
dead_time = 1.3 * tau
dead_step_output = np.where(t >= dead_time, 1, 0)

# Recirculation Output Response
recirculation_step_output = 0.5 + 0.5 * np.sin(t/tau * 2 * np.pi) * np.exp(-t/tau)

# Visualization
plt.figure(figsize=(15, 12))

reactors = [("Plug Flow", pf_step_output),
            ("CSTR", cstr_step_output),
            ("Bypass", bypass_step_output),
            ("Dead Volume", dead_step_output),
            ("Recirculation", recirculation_step_output)]

for i, (name, step_out) in enumerate(reactors):
    plt.subplot(2, 3, i+1)
    plt.plot(t, step_input, label="Step Input", linestyle='--')
    plt.plot(t, step_out, label=f"{name} Step Output")
    plt.title(name)
    plt.legend()

plt.tight_layout()
plt.show()


# Time array
t = np.linspace(0, 10, 1000)
tau = 5  # space time for the CSTR part

# Step Input Tracer Concentration
step_input = np.where(t > 0, 1, 0)

# CSTR without Dead Volume Output Response
cstr_step_output = 1 - np.exp(-t/tau)

# CSTR with Dead Volume Output Response (assuming a 30% delay)
dead_time = 1.3 * tau
delayed_step_input = np.where(t > dead_time, 1, 0)
cstr_with_dead_step_output = 1 - np.exp(-(t-dead_time)/tau)
cstr_with_dead_step_output[t < dead_time] = 0

# Visualization
plt.figure(figsize=(10, 6))

plt.plot(t, step_input, label="Step Input", linestyle='--')
plt.plot(t, cstr_step_output, label="CSTR Output")
plt.plot(t, cstr_with_dead_step_output, label="CSTR with Dead Volume Output")
plt.title("CSTR Step Responses")
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Time array
t = np.linspace(0, 10, 1000)
tau = 5  # space time

# Pulse Input Tracer Concentration
pulse_input = np.where(t < 0.1, 1, 0)

# Plug Flow Output Response
pf_pulse_output = np.where(t == tau, 1, 0)

# CSTR Output Response
cstr_pulse_output = np.convolve(pulse_input, np.exp(-t/tau), mode='same')

# CSTR with Dead Volume Output Response (assuming a 30% delay)
dead_time = 1.3 * tau
delayed_pulse_input = np.roll(pulse_input, int(dead_time*(len(t)-1)/t[-1]))
cstr_dead_pulse_output = np.convolve(delayed_pulse_input, np.exp(-t/tau), mode='same')

# Bypass Output Response (assuming 30% bypass)
bypass_fraction = 0.3
bypass_pulse_output = bypass_fraction * np.where(t < 0.1, 1, 0) + (1 - bypass_fraction) * np.where(t == tau, 1, 0)

# Dead Volume Output Response (pure delay, 30%)
dead_pulse_output = np.where(t == dead_time, 1, 0)

# Recirculation Output Response
recirculation_pulse_output = np.sin(t/tau * 2 * np.pi) * np.exp(-t/tau)

# Visualization
plt.figure(figsize=(15, 12))

reactors = [("Plug Flow", pf_pulse_output),
            ("CSTR", cstr_pulse_output),
            ("CSTR with Dead Volume", cstr_dead_pulse_output),
            ("Bypass", bypass_pulse_output),
            ("Pure Dead Volume", dead_pulse_output),
            ("Recirculation", recirculation_pulse_output)]

for i, (name, pulse_out) in enumerate(reactors):
    plt.subplot(2, 3, i+1)
    plt.plot(t, pulse_input, label="Pulse Input", linestyle='--')
    plt.plot(t, pulse_out, label=f"{name} Pulse Output")
    plt.title(name)
    plt.legend()

plt.tight_layout()
plt.show()
