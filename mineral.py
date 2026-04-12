import streamlit as st
import numpy as np
from scipy.optimize import fsolve
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(page_title="REE Optimizer - Group 6", layout="wide", page_icon="⛏️")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("⛏️ REE In-Situ Leaching Optimization Dashboard")
st.markdown("**Powered by UMK Jeli Geoscience Data (Fendy & Ismail) | Internal Diffusion SCM**")
st.markdown("---")

# ==========================================
# 2. MATHEMATICAL ENGINE (UMK CALIBRATED)
# ==========================================
def scm_equation(X, k, t_min):
    X_safe = np.clip(X, 0, 0.9999) 
    return 1 - (2/3)*X_safe - (1 - X_safe)**(2/3) - (k * t_min)

def calculate_yield(molarity, time_hours, salt_type):
    t_min = time_hours * 60
    
    # Constants derived from Table 1 (Sample GM48) of the UMK Journal
    # Calibrated to 0.5M experimental baseline
    if salt_type == "Ammonium Nitrate (NH4NO3)":
        k_base = 0.00101  # 61.47% recovery @ 1hr
    elif salt_type == "Ammonium Sulfate ((NH4)2SO4)":
        k_base = 0.00053  # 46.52% recovery @ 1hr
    else:
        k_base = 0.00043  # 42.67% recovery @ 1hr
        
    k_adjusted = k_base * (molarity / 0.5)
    
    sol, info, ier, msg = fsolve(scm_equation, 0.5, args=(k_adjusted, t_min), full_output=True)
    return np.clip(sol[0], 0, 1) * 100 if ier == 1 else 100.0

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("⚙️ Operational Parameters")
    salt_type = st.selectbox(
        "Select Lixiviant (Salt):",
        ["Ammonium Sulfate ((NH4)2SO4)", "Ammonium Nitrate (NH4NO3)", "Magnesium Chloride (MgCl)"]
    )
    
    target_yield = st.slider("Target Recovery (%)", 30, 99, 85)
    
    st.markdown("---")
    st.subheader("Constraints")
    max_time = st.slider("Max Duration (Hours)", 24, 240, 120)
    max_molarity = st.slider("Max Concentration (M)", 0.1, 2.0, 1.0, 0.1)
    
    st.info(f"**Data Source:** Sample GM48 (Saprolite Horizon). Calibrated for {salt_type}.")

# ==========================================
# 4. SIMULATION EXECUTION
# ==========================================
# Generate high-resolution matrix (2,500 data points)
time_grid = np.linspace(1, max_time, 50)
molarity_grid = np.linspace(0.1, max_molarity, 50)
T, M = np.meshgrid(time_grid, molarity_grid)
Z = np.zeros_like(T)

opt_time, opt_molarity = None, None
min_cost = float('inf')

for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        y = calculate_yield(M[i,j], T[i,j], salt_type)
        Z[i,j] = y
        if y >= target_yield:
            # Cost function: (Molarity * 100) + (Time * 1)
            current_cost = (M[i,j] * 100) + (T[i,j] * 1)
            if current_cost < min_cost:
                min_cost = current_cost
                opt_time, opt_molarity = T[i,j], M[i,j]

# ==========================================
# 5. MAIN DASHBOARD DISPLAY
# ==========================================
st.subheader(f"📊 Optimization Results for {salt_type}")

if opt_time:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimum Molarity", f"{opt_molarity:.2f} M")
    c2.metric("Optimum Time", f"{opt_time:.1f} Hours")
    c3.metric("Predicted Yield", f"{target_yield}%")
    c4.metric("Cost Score", f"{min_cost:.1f}")
else:
    st.warning("⚠️ Target recovery not reachable within current constraints. Please increase Time or Molarity.")

st.markdown("---")

# TABS FOR DUAL VISUALIZATION
tab1, tab2 = st.tabs(["📈 3D Surface View", "🗺️ 2D Contour Map"])

with tab1:
    st.write("Visualize the overall extraction landscape.")
    fig3d = go.Figure(data=[go.Surface(z=Z, x=time_grid, y=molarity_grid, colorscale='Viridis')])
    if opt_time:
        fig3d.add_trace(go.Scatter3d(x=[opt_time], y=[opt_molarity], z=[target_yield], 
                                     mode='markers', marker=dict(size=8, color='red'), name='Sweet Spot'))
    fig3d.update_layout(scene=dict(xaxis_title='Time (h)', yaxis_title='Molarity (M)', zaxis_title='Yield (%)'),
                        height=600, margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig3d, use_container_width=True)

with tab2:
    st.write("Precise analytical view for operational decision making.")
    fig2d = go.Figure(data=go.Contour(z=Z, x=time_grid, y=molarity_grid, colorscale='Viridis',
                                     contours=dict(showlabels=True, labelfont=dict(size=12, color='white'))))
    if opt_time:
        fig2d.add_trace(go.Scatter(x=[opt_time], y=[opt_molarity], mode='markers+text', 
                                   marker=dict(size=15, color='red', symbol='x'), 
                                   text=["Optimum"], textposition="top center"))
    fig2d.update_layout(xaxis_title='Time (Hours)', yaxis_title='Molarity (M)', height=600)
    st.plotly_chart(fig2d, use_container_width=True)

# ==========================================
# 6. LITERATURE REFERENCE SECTION
# ==========================================
with st.expander("📖 View Scientific References"):
    st.write("""
    1. **Fendy, N. A., & Ismail, R.** (UMK Jeli). *Leaching of NR-REE from IAC using Monovalent Salt Solution*. 
       Base data used for Sample GM48 calibration.
    2. **Shrinking Core Model (SCM)** - Internal Diffusion Control formula used for kinetic simulation.
    3. **Optimization Logic:** The 'Sweet Spot' is defined as the minimum concentration and time required to cross the target yield threshold.
    """)