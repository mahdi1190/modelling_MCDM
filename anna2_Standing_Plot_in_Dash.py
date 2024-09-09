import os
import numpy as np
import pickle
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
from scipy.optimize import fsolve

# File to store the current run number
run_number_file = 'run_number.pkl'

# Load previous run number or initialize to 1
if os.path.exists(run_number_file):
    with open(run_number_file, 'rb') as f:
        current_run_number = pickle.load(f) + 1
else:
    current_run_number = 1

# Save the updated run number
with open(run_number_file, 'wb') as f:
    pickle.dump(current_run_number, f)

# Define a set of colors (default Plotly colors)
colors = ["#000000",
    "#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", 
    "#5C1379", "#FF6692", "#44AA99"
]
#color = colors[(current_run_number - 1) % len(colors)]

# Initialize parameters
def init_params():
    global k1, KM, KS, Kp, Kes1, Kes2, E, l, a, r, f, Pu, Pn, Pc, Paa
    global k2, k2r, k3, k3r, k4, k4r, k5, k5r, k6, k6r, k7, k7r
    global S, V, P_h, P_k, P_Na, P_NH4, ec, C0, kTe, Ko, NA, B, alpha_val, tol

    l = 3.0
    a = 1
    k1 = 0.00022
    E = 150
    r = 82e-7
    f = 5e-3
    KM = 0.003
    KS = 3
    Kp = 2
    Kes1 = 5e-6
    Kes2 = 2e-9
    k2 = 1440
    k2r = 2.58e12
    k3 = 2.22
    k3r = 4.74e6
    k4 = 168
    k4r = 3e12
    k5 = 6e-2
    k5r = 6e12
    k6 = 60
    k6r = 1.5e9
    k7r = 2.7e12
    AA_Ka = 0.000017378
    k7 = k7r * AA_Ka
    R = r / 10
    NA = 6.0221412927e23
    ec = 1.6021766e-19
    kTe = 25.69
    V = 4 / 3 * np.pi * R**3
    S = 4 * np.pi * r**2
    tol = 1e-12

    Pu = 8.48E-05
    Pn = 1e-1
    Pc = 6.00E-03
    Paa = 6.00E-04
    C0 = 1 * 10**(-9)
    P_h = 0
    P_k = 0
    P_Na = 0
    P_NH4 = 0
    alpha_val = ec * NA / C0 / kTe
    Ko = tol
    B = 1e-10

    # Initial conditions vector
    y0 = np.array([
        tol, tol, tol, tol, tol, tol, 1e-5, 1e-9, 5e-5, tol, 0, 0, tol, 
        (0.025 * AA_Ka) / (1e-5 + AA_Ka), 0.025 - (0.025 * AA_Ka) / (1e-5 + AA_Ka), 
        0.025, tol, 0, tol, tol, tol, 1e-5, 1e-9, (0.025 * AA_Ka) / (1e-5 + AA_Ka), 
        0.025 - (0.025 * AA_Ka) / (1e-5 + AA_Ka), 0.1, tol, 0, 0, 0
    ])
    
    return y0

# ODE system
def ureaserate2(t, y, yd):
    global k1, KM, KS, Kp, Kes1, Kes2, E, l, a, r, f, Pu, Pn, Pc, Paa
    global k2, k2r, k3, k3r, k4, k4r, k5, k5r, k6, k6r, k7, k7r
    global S, V, P_h, P_k, P_Na, P_NH4, ec, C0, kTe, Ko, NA, B, alpha_val

    Ur, NH3, NH4, CO2, HCO3, CO3, H, OH, pyOH, pyO, _, _, hnum, CH3COO, CH3COOH, Uro, NH3o, NH4o, CO2o, HCO3o, CO3o, Ho, OHo, CH3COOo, CH3COOHo, K, knum, _, _, NH4num = y

    u = alpha_val / S * ((hnum + knum) / NA - B * V)
    
    R1 = k1 * E * Ur / ((1 + Kes2 / H + H / Kes1) * (KM + Ur * (1 + Ur / KS)) * (1 + NH4 / Kp))

    dydt = np.zeros_like(y)
    
    dydt[0] = -R1 + l * a * Pu / r * (Uro - Ur)
    dydt[1] = 2 * R1 + k2 * NH4 - k2r * NH3 * H + l * a * Pn / r * (NH3o - NH3)
    dydt[2] = -k2 * NH4 + k2r * NH3 * H - P_NH4 * S * perm(NH4, NH4o, u, NA) / (NA * V)
    dydt[3] = R1 - k3 * CO2 + k3r * H * HCO3 + l * a * Pc / r * (CO2o - CO2)
    dydt[4] = k3 * CO2 - k3r * HCO3 * H - k4 * HCO3 + k4r * CO3 * H
    dydt[5] = k4 * HCO3 - k4r * CO3 * H
    dydt[6] = k2 * NH4 - k2r * NH3 * H + k4 * HCO3 - k4r * CO3 * H + k5 - k5r * H * OH + k3 * CO2 - k3r * HCO3 * H + k6 * pyOH - k6r * H * pyO + k7 * CH3COOH - k7r * CH3COO * H - P_h * S * perm(H, Ho, u, NA) / (NA * V)
    dydt[7] = k5 - k5r * H * OH
    dydt[8] = -k6 * pyOH + k6r * H * pyO
    dydt[9] = k6 * pyOH - k6r * H * pyO
    dydt[12] = -P_h * S * perm(H, Ho, u, NA)
    dydt[13] = k7 * CH3COOH - k7r * CH3COO * H
    dydt[14] = -k7 * CH3COOH + k7r * CH3COO * H + l * a * Paa / r * (CH3COOHo - CH3COOH)
    dydt[15] = -(3 * a * Pu / r) * f * (Uro - Ur)
    dydt[16] = k2 * NH4o - k2r * NH3o * Ho - (l * a * Pn / r) * f * (NH3o - NH3)
    dydt[17] = -k2 * NH4o + k2r * NH3o * Ho + f * P_NH4 * S * perm(NH4, NH4o, u, NA) / (NA * V)
    dydt[18] = -k3 * CO2o + k3r * Ho * HCO3o - (l * a * Pc / r) * f * (CO2o - CO2)
    dydt[19] = k3 * CO2o - k3r * HCO3o * Ho - k4 * HCO3o + k4r * CO3o * Ho
    dydt[20] = k4 * HCO3o - k4r * CO3o * Ho
    dydt[21] = k2 * NH4o - k2r * NH3o * Ho + k4 * HCO3o - k4r * CO3o * Ho + k5 - k5r * Ho * OHo + k3 * CO2o - k3r * HCO3o * Ho + k7 * CH3COOHo - k7r * CH3COOo * Ho + f * P_h * S * perm(H, Ho, u, NA) / (NA * V)
    dydt[22] = k5 - k5r * Ho * OHo
    dydt[23] = k7 * CH3COOHo - k7r * CH3COOo * Ho
    dydt[24] = -k7 * CH3COOHo + k7r * CH3COOo * Ho - (l * a * Paa / r) * f * (CH3COOHo - CH3COOH)
    dydt[25] = -P_k * S * perm(K, Ko, u, NA) / (V * NA)
    dydt[26] = -P_k * S * perm(K, Ko, u, NA)
    dydt[29] = -P_NH4 * S * perm(NH4, NH4o, u, NA)
    return dydt - yd

# Permeability calculation
def perm(a, b, u, NA):
    if abs(u) > 0:
        y = u * (a - b * np.exp(-u)) / (1 - np.exp(-u))
    else:
        y = a - b
    return y * 1e-3 * NA

# Initial derivative calculation
def compute_initial_derivatives(y0):
    def residual(yd):
        return ureaserate2(0, y0, yd)
    yd0, infodict, ier, msg = fsolve(residual, np.zeros_like(y0), full_output=True)
    if ier != 1:
        raise RuntimeError(f"Initial derivative calculation did not converge: {msg}")
    return yd0

def run_model():
    y0 = init_params()
    yd0 = compute_initial_derivatives(y0)
    
    problem = Implicit_Problem(ureaserate2, y0, yd0, 0)
    solver = IDA(problem)
    
    solver.atol = 1e-21
    solver.rtol = 1e-6
    tf = 300
    t, y, _ = solver.simulate(tf, tf*10)
  
    V_mem = alpha_val / S * ((y[:, 12] + y[:, 26] + y[:, 29]) / NA - B * V) * kTe
    pH = -np.log(y[:, 6]) / np.log(10)
    pHo = -np.log(y[:, 21]) / np.log(10)

    # Find the index where pH first exceeds or equals 6.4
    index_pH_6_4 = np.where(pH >= 6.4)[0]

    if index_pH_6_4.size > 0:
        time_pH_6_4_inside_index = t[index_pH_6_4[0]]
        print(f"The time when pH first exceeds or equals 6.4 is: {time_pH_6_4_inside_index} minutes")
    else:
        print("pH never exceeds or equals 6.4 in the simulation.")

    return t, y, V_mem, pH, pHo

def generate_figure():
    t, y, V_mem, pH, pHo = run_model()

    fig = make_subplots(rows=3, cols=3, subplot_titles=(
        'pH', '[Urea] / M', 'V<sub>mem</sub>',
        '[Ammonia] / M', '[Ammonium] / M', 'Excess H+ in',
        '[CH3COO-] / M', '[CH3COOH] / M', 'Excess K+ in'
    ))

    # Subplot 1: pH
    fig.add_trace(go.Scatter(x=t, y=pH, mode='lines', name='pH in', line=dict(color=colors[0])), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=pHo, mode='lines', name='pH out', line=dict(color=colors[0], dash='dash')), row=1, col=1)
    fig.update_xaxes(title_text="Time / min", row=1, col=1)
    fig.update_yaxes(title_text="pH", row=1, col=1)

    # Subplot 2: U / M
    fig.add_trace(go.Scatter(x=t, y=y[:, 0], mode='lines', name='[Urea] in', line=dict(color=colors[1])), row=1, col=2)
    fig.update_xaxes(title_text="Time / min", row=1, col=2)
    #fig.update_yaxes(title_text="U / M", row=1, col=2)

    # Subplot 3: Membrane potential
    fig.add_trace(go.Scatter(x=t, y=V_mem, mode='lines', name='V<sub>mem</sub>', line=dict(color=colors[2])), row=1, col=3)
    fig.update_xaxes(title_text="Time / min", row=1, col=3)
    #fig.update_yaxes(title_text=r'$V_{\mathrm{mem}}$ / mV', row=1, col=3)

    # Subplot 4: Ammonia in and out
    fig.add_trace(go.Scatter(x=t, y=y[:, 1], mode='lines', name='[Ammonia] in', line=dict(color=colors[3])), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=y[:, 16], mode='lines', name='[Ammonia] out', line=dict(color=colors[3], dash='dash')), row=2, col=1)
    fig.update_xaxes(title_text="Time / min", row=2, col=1)
    #fig.update_yaxes(title_text="Ammonia / M", row=2, col=1)

    # Subplot 5: Ammonium in and out
    fig.add_trace(go.Scatter(x=t, y=y[:, 2], mode='lines', name='[Ammonium] in ', line=dict(color=colors[4])), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=y[:, 17], mode='lines', name='[Ammonium] out', line=dict(color=colors[4], dash='dash')), row=2, col=2)
    fig.update_xaxes(title_text="Time / min", row=2, col=2)
    #fig.update_yaxes(title_text="Ammonium / M", row=2, col=2)

    # Subplot 6: Excess H+
    fig.add_trace(go.Scatter(x=t, y=y[:, 12], mode='lines', name='Excess H+ ions in', line=dict(color=colors[5])), row=2, col=3)
    fig.update_xaxes(title_text="Time / min", row=2, col=3)
    #fig.update_yaxes(title_text="Excess H+ ions", row=2, col=3)

    # Subplot 7: CH3COO- anion
    fig.add_trace(go.Scatter(x=t, y=y[:, 13], mode='lines', name='[CH3COO-]', line=dict(color=colors[6])), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=y[:, 23], mode='lines', name='[CH3COO-]', line=dict(color=colors[6], dash='dash')), row=3, col=1)
    fig.update_xaxes(title_text="Time / min", row=3, col=1)
    #fig.update_yaxes(title_text="CH3COO- / M", row=3, col=1)

    # Subplot 8: CH3COOH
    fig.add_trace(go.Scatter(x=t, y=y[:, 14], mode='lines', name='[CH3COOH]', line=dict(color=colors[7])), row=3, col=2)
    fig.add_trace(go.Scatter(x=t, y=y[:, 24], mode='lines', name='[CH3COOH] out', line=dict(color=colors[7], dash='dash')), row=3, col=2)
    fig.update_xaxes(title_text="Time / min", row=3, col=2)
    #fig.update_yaxes(title_text="CH3COOH / M", row=3, col=2)

    # Subplot 9: Excess K+
    fig.add_trace(go.Scatter(x=t, y=y[:, 26], mode='lines', name='Excess K+ ions in', line=dict(color=colors[8])), row=3, col=3)
    fig.update_xaxes(title_text="Time / min", row=3, col=3)
    #fig.update_yaxes(title_text="Excess K+ ions", row=3, col=3)

    fig.update_layout(height=800, width=1200, title_text="Dynamic Plot from Monitored File", showlegend=True)
    
    return fig

# Dash app setup
app = dash.Dash(__name__)

# Initial layout of the app
app.layout = html.Div([
    #html.H1("Dynamic Plot Update"),
    dcc.Slider(id='slider', min=1100, max=2000, step=100, value=1300,
               marks={x: str(x) for x in [1200, 1400, 1600, 1800]}),
    dcc.Graph(id='live-graph', style={'height': '150vh'}),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    )
])

@app.callback(
    Output('live-graph', 'figure'),
    Input('slider', 'value'),
    Input('interval-component', 'n_intervals')
)
def update_graph_live(width, n):
    height = int(0.7 * width)
    fig = generate_figure()
    fig.update_layout(width=int(width), height=height)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
