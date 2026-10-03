import numpy as np
import matplotlib.pyplot as plt

def calculate_ntp_performance(T_c, P_c, P_e, P_a, A_throat, A_exit, gamma, R, g0):

    # 1. Exhaust Velocity (v_e)
    term1 = (2 * gamma * R * T_c) / (gamma - 1)
    term2 = 1 - (P_e / P_c)**((gamma - 1) / gamma)
    v_e = np.sqrt(term1 * term2)
    
    # 2. Choked Mass Flow Rate (m_dot)
    choke_term = np.sqrt((gamma / R) * (2 / (gamma + 1))**((gamma + 1) / (gamma - 1)))
    m_dot = (A_throat * P_c / np.sqrt(T_c)) * choke_term
    
    # 3. Total Thrust (F)
    thrust = (m_dot * v_e) + ((P_e - P_a) * A_exit)
    
    # 4. Specific Impulse (Isp)
    Isp = v_e / g0
    
    return thrust, Isp, m_dot, v_e

# Properties and Assumptions
gamma = 1.4           # Specific heat ratio for Hydrogen
R = 4124.0            # Gas constant for Hydrogen (J/kg*K)
g0 = 9.81             # Gravity at sea level (m/s^2)

# Engine Geometry (NERVA-style 24:1 Expansion)
A_throat = 0.05       # Throat cross-sectional area (m^2)
A_exit = 1.2          # Exit cross-sectional area (m^2)

# Pressure Conditions
P_c = 6.89e6          # Chamber Pressure (~1000 psi to Pascals)
P_e = 20000.0         # Nozzle Exit Pressure (Pa)
P_a = 0.0             # Ambient Pressure (Vacuum of space)


# Simulating the reactor heating from a low idle (1000 K) up to extreme thermal limits (3200 K)
T_c_array = np.linspace(1000, 3200, 100)

# Initialize storage arrays
thrust_array = np.zeros(len(T_c_array))
isp_array = np.zeros(len(T_c_array))
mdot_array = np.zeros(len(T_c_array))

# Run the physics solver for every temperature
for i, T_c in enumerate(T_c_array):
    thrust, Isp, m_dot, v_e = calculate_ntp_performance(T_c, P_c, P_e, P_a, A_throat, A_exit, gamma, R, g0)
    
    # Thrust to KN
    thrust_array[i] = thrust / 1000.0
    isp_array[i] = Isp
    mdot_array[i] = m_dot

# Visualized Data
fig, ax1 = plt.subplots(figsize=(10, 6))


color1 = 'tab:red'
ax1.set_xlabel('Reactor Core Temperature (K)', fontweight='bold')
ax1.set_ylabel('Total Thrust (kN)', color=color1, fontweight='bold')
ax1.plot(T_c_array, thrust_array, color=color1, linewidth=2.5, label='Thrust (kN)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  
color2 = 'tab:blue'
ax2.set_ylabel('Specific Impulse - Isp (Seconds)', color=color2, fontweight='bold')  
ax2.plot(T_c_array, isp_array, color=color2, linewidth=2.5, linestyle='--', label='Specific Impulse (s)')
ax2.tick_params(axis='y', labelcolor=color2)

ax1.axvline(x=2500, color='black', linestyle=':', alpha=0.7)
ax1.text(2520, min(thrust_array) + 20, '', rotation=90, fontsize=10, alpha=0.8)

plt.title('Nuclear Thermal Propulsion Performance vs. Reactor Temperature', fontweight='bold', fontsize=14)
fig.tight_layout()  
plt.show()

print(f"--- 2500 K Operating Point Metrics ---")
idx_2500 = np.abs(T_c_array - 2500).argmin()
print(f"Thrust: {thrust_array[idx_2500]:.2f} kN")
print(f"Specific Impulse (Isp): {isp_array[idx_2500]:.2f} Seconds")
print(f"Mass Flow Rate: {mdot_array[idx_2500]:.2f} kg/s")