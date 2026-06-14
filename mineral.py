import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
import time
from fpdf import FPDF
import base64
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & INTERACTIVE CSS INJECTION
# ==========================================
st.set_page_config(layout="wide", page_title="ISL REE Optimization Model - Group 6", page_icon="💎")

st.markdown("""
<style>
    .stMetric {
        background-color: white;
        padding: 1.2rem;
        border-radius: 1.2rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border: 1px solid rgba(59, 130, 246, 0.2);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        cursor: pointer;
    }
    .stMetric:hover {
        border-color: #3B82F6;
        box-shadow: 0 12px 20px rgba(59, 130, 246, 0.15);
        transform: translateY(-4px);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .autopilot-box {
        background: linear-gradient(135deg, #F0F9FF 0%, #DBEAFE 100%);
        border-left: 6px solid #2563EB;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .autopilot-text {
        color: #1E3A8A;
        font-size: 15px;
        font-weight: 500;
        margin: 0;
    }
    
    /* SCADA INDUSTRIAL CRITICAL ALARM */
    @keyframes scada-flash {
        0% { background-color: #7f1d1d; box-shadow: 0 0 10px rgba(220, 38, 38, 0.5); }
        50% { background-color: #dc2626; box-shadow: 0 0 25px rgba(220, 38, 38, 0.9); }
        100% { background-color: #7f1d1d; box-shadow: 0 0 10px rgba(220, 38, 38, 0.5); }
    }
    .scada-alarm-container {
        border-left: 12px solid #000000;
        border-right: 2px solid #ef4444;
        border-top: 2px solid #ef4444;
        border-bottom: 2px solid #ef4444;
        padding: 18px 20px;
        border-radius: 4px;
        color: #ffffff;
        font-family: 'Courier New', Courier, monospace;
        animation: scada-flash 1.2s infinite;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .scada-header {
        font-size: 18px;
        font-weight: 900;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        color: #ffffff;
        text-transform: uppercase;
    }
    .scada-body {
        font-size: 15px;
        font-weight: bold;
        color: #fca5a5;
        line-height: 1.4;
    }
    .scada-code {
        background-color: #000000;
        color: #ef4444;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 13px;
        margin-right: 8px;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Session State untuk kawal sistem Toast 
if 'prev_molarity' not in st.session_state:
    st.session_state.prev_molarity = 1.60
if 'prev_time' not in st.session_state:
    st.session_state.prev_time = 144.0

# ==========================================
# 2. SIDEBAR & GEOLOGICAL PROFILE
# ==========================================
st.sidebar.markdown("## 🎛️ DSS CONTROL PANEL")
st.sidebar.caption("Adjust baseline parameters to run live simulations.")

with st.sidebar.container(border=True):
    st.markdown("#### ⚙️ Operational Limits")
    max_time = st.slider("Max Pumping Time (Hours)", 24, 300, 120, 12, help="Hardware constraint for pump lifespan.")
    max_molarity = st.slider("Max Concentration (M)", 0.2, 3.0, 2.0, 0.1, help="Safety limit for Ammonium Sulphate.")

with st.sidebar.container(border=True):
    st.markdown("#### 🎯 Economic Target")
    target_yield = st.slider("Target Recovery Yield (%)", 50, 100, 70, help="Minimum yield required for company profitability.")

with st.sidebar.container(border=True):
    st.markdown("#### 🌍 Geological Profile")
    ree_content = st.number_input("Input Site REE Grade (g/ton):", min_value=0, max_value=2000, value=350, step=10)
    
    if ree_content > 400:
        status_label, status_desc, status_color, text_color = "ECONOMIC MINING", "High-grade ore. Ideal for full-scale operations.", "rgba(46, 204, 113, 0.2)", "#2ECC71"
    elif 300 <= ree_content <= 400:
        status_label, status_desc, status_color, text_color = "POTENTIAL MINING", "Feasible. Monitor OPEX closely.", "rgba(241, 196, 15, 0.2)", "#F1C40F"
    elif 100 <= ree_content < 300:
        status_label, status_desc, status_color, text_color = "POSSIBLE MINING", "Marginal. Requires optimized recovery.", "rgba(52, 152, 219, 0.2)", "#3498DB"
    else:
        status_label, status_desc, status_color, text_color = "NOT FEASIBLE", "Grade too low. Uneconomical for ISL.", "rgba(231, 76, 60, 0.2)", "#E74C3C"

    st.markdown(f"""
        <div style="background-color:{status_color}; padding:12px; border-radius:8px; border:1.5px solid {text_color}; text-align:center; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);">
            <h5 style="color:{text_color}; margin:0px; font-weight:800; letter-spacing: 0.5px;">{status_label}</h5>
            <p style="font-size:12px; margin:4px 0px 0px 0px; color:white;">{status_desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎓 Project Developers")
st.sidebar.info(
    "**Group 6 (Mineral Technology)**\n"
    "• Muhammad Amir Bin Nasrudin\n"
    "• Nur Irdina Syakila Binti Mohamed Noor\n"
    "• Ayu Aneesha Binti Abd Halim\n"
    "• Thiviyadharshini A/P Mani Rajah\n\n"
    "**Supervisor:**\n"
    "Assoc. Prof. Ts. ChM. Dr Abdul Hafidz Bin Yusoff"
)

# ==========================================
# 3. BACKGROUND DATA ENGINE
# ==========================================
data_points = [
    {'Time': 1, 'Molarity': 0.5, 'Recovery': 46.5, 'Source': 'Fendy & Ismail (GM48 0.5M)'},
    {'Time': 24, 'Molarity': 1.5, 'Recovery': 15.0, 'Source': 'Miiro 2023'},
    {'Time': 72, 'Molarity': 1.5, 'Recovery': 31.0, 'Source': 'Miiro 2023'},
    {'Time': 144, 'Molarity': 1.5, 'Recovery': 50.0, 'Source': 'Miiro 2023'},
    {'Time': 216, 'Molarity': 1.5, 'Recovery': 60.0, 'Source': 'Miiro 2023'},
    {'Time': 288, 'Molarity': 1.5, 'Recovery': 69.0, 'Source': 'Miiro 2023'},
    {'Time': 24, 'Molarity': 0.2, 'Recovery': 30.0, 'Source': 'He et al. 2016'},
    {'Time': 72, 'Molarity': 0.2, 'Recovery': 42.0, 'Source': 'He et al. 2016'}
]
journal_data = pd.DataFrame(data_points)

C_INTERCEPT = 12.0    
M_MOLARITY = 20.0     
M_TIME = 0.15          

def calc_y_mx_c(molarity, time):
    return max(C_INTERCEPT + (M_MOLARITY * molarity) + (M_TIME * time), 0.0) 

# ==========================================
# 4. EXPORT GENERATORS (EXCEL & PDF)
# ==========================================
@st.cache_data
def generate_excel(m_max, y_target, t_max):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        journal_data.to_excel(writer, sheet_name='Background Validation', index=False)
        m_range_excel = np.linspace(0.1, m_max, 50)
        t_needed_excel = [max(0, min((y_target - C_INTERCEPT - (M_MOLARITY * m)) / M_TIME, t_max)) for m in m_range_excel]
        df_tradeoff = pd.DataFrame({'Chemical Concentration (M)': m_range_excel, 'Required Time (Hours)': t_needed_excel})
        df_tradeoff.to_excel(writer, sheet_name='Optimization Boundary', index=False)
    return output.getvalue()

def generate_pdf_report(molarity, hours, yield_val, target_val, profit, opex, status):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 10, "ISL DECISION SUPPORT SYSTEM - EXECUTIVE SUMMARY", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. OPERATIONAL PARAMETERS", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 6, f"- Ammonium Sulphate Concentration: {molarity:.2f} M", ln=True)
    pdf.cell(0, 6, f"- Total Pumping Time: {hours:.1f} Hours", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. SYSTEM YIELD & ESG COMPLIANCE", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 6, f"- Target Yield Required: {target_val:.1f}%", ln=True)
    pdf.cell(0, 6, f"- Predicted Recovery Yield: {yield_val:.1f}%", ln=True)
    
    if molarity > 2.0:
        pdf.set_text_color(220, 38, 38)
        pdf.cell(0, 6, "- ESG STATUS: CRITICAL WARNING (High Contamination Risk)", ln=True)
    else:
        pdf.set_text_color(16, 185, 129)
        pdf.cell(0, 6, "- ESG STATUS: NOMINAL (Safe Environmental Operation)", ln=True)
        
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. ECONOMIC FEASIBILITY (100-TON PILOT WELL)", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 6, f"- Estimated OPEX: RM {opex:,.2f}", ln=True)
    pdf.cell(0, 6, f"- Projected Net Profit: RM {profit:,.2f}", ln=True)
    pdf.cell(0, 6, f"- Economic Status: {status}", ln=True)
    
    return pdf.output(dest='S').encode('latin1')

st.sidebar.markdown("---")
# Dual Export Buttons
export_col1, export_col2 = st.sidebar.columns(2)
with export_col1:
    st.download_button("📊 EXCEL DATA", data=generate_excel(max_molarity, target_yield, max_time), file_name="REE_ISL_Data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

# ==========================================
# 5. MAIN PANEL DISPLAY & INTERACTIVE LOGIC
# ==========================================

st.markdown("""
<div style="background: linear-gradient(135deg, #1E3A8A, #3B82F6, #2563EB); padding: 30px; border-radius: 12px; text-align: center; margin-bottom: 25px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);">
    <h1 style="color: #ffffff; margin: 0; font-weight: 900; letter-spacing: 1px; font-size: 2.2rem;">In-Situ Leaching (ISL) Decision Support System</h1>
    <p style="color: #DBEAFE; margin: 8px 0 0 0; font-size: 1.1rem; font-weight: 500;">Optimization Model for Rare Earth Elements (REE) Using Ammonium Sulphate</p>
</div>
""", unsafe_allow_html=True)

# --- LIVE PREDICTOR CARD ---
with st.container(border=True):
    st.markdown("<h3 style='color: #1E293B;'>🧮 Step 2: Live Yield & OPEX Predictor</h3>", unsafe_allow_html=True)
    
    input_col1, input_col2, input_col3 = st.columns(3)
    with input_col1:
        user_molarity = st.number_input("Ammonium Sulphate Concentration (M):", min_value=0.0, max_value=5.0, value=1.60, step=0.05, format="%.2f")
    with input_col2:
        user_time = st.number_input("Pumping Time (Hours):", min_value=0.0, max_value=500.0, value=144.0, step=12.0, format="%.1f")
    
    live_predicted_yield = calc_y_mx_c(user_molarity, user_time)
    clamped_display_yield = min(live_predicted_yield, 100.0) 
    
    # OPEX Breakdown
    base_cost = 500.0
    chemical_cost = user_molarity * 1000.0
    electricity_cost = user_time * 12.0
    est_opex = base_cost + chemical_cost + electricity_cost

    with input_col3:
        st.info(f"💰 **Est. OPEX (Pilot Well):**\n### RM {est_opex:,.2f}")

    if st.session_state.prev_molarity != user_molarity or st.session_state.prev_time != user_time:
        with st.spinner("Recalculating Kinetics & ROI..."):
            time.sleep(0.3) 
        if user_molarity > 2.0:
            st.toast("🚨 ESG SYSTEM INTERLOCK TRIGGERED!", icon="☢️")
        elif clamped_display_yield >= target_yield:
            st.toast("✅ Optimization Complete: Target Achieved.", icon="🟢")
        else:
            st.toast("⚙️ Parameters Updated. Target NOT met.", icon="⚠️")
        st.session_state.prev_molarity = user_molarity
        st.session_state.prev_time = user_time

    st.markdown("---")
    
    yield_color = "#10B981" if clamped_display_yield >= target_yield else "#F59E0B"
    status_text = "OPTIMAL (TARGET ACHIEVED)" if clamped_display_yield >= target_yield else "SUB-OPTIMAL (BELOW TARGET)"
    
    st.markdown(f"""
    <div style="margin-top: 10px; margin-bottom: 25px; padding: 15px; border-radius: 8px; background-color: #F8FAFC; border: 1px solid #E2E8F0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-weight: 600; color: #475569; font-size: 14px;">System Recovery Yield vs Target ({target_yield}%)</span>
            <span style="font-weight: 800; color: {yield_color}; font-size: 15px;">{clamped_display_yield:.1f}% - {status_text}</span>
        </div>
        <div style="width: 100%; background-color: #E2E8F0; border-radius: 6px; height: 16px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);">
            <div style="width: {clamped_display_yield}%; background-color: {yield_color}; height: 100%; transition: width 0.5s ease-in-out, background-color 0.5s ease-in-out;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if user_molarity > 2.0:
        st.markdown(f"""
        <div class="scada-alarm-container">
            <div class="scada-header">☢️ CRITICAL ALARM: SYSTEM SAFETY INTERLOCK TRIGGERED</div>
            <div class="scada-body">
                <span class="scada-code">ERR-ESG-901</span> Chemical concentration ({user_molarity:.2f} M) exceeds structural safety limits.<br>
                <span style="color:#ffffff;">RISK:</span> Severe ammonia-nitrogen groundwater leaching and soil degradation.<br>
                <span style="color:#ffffff;">ACTION:</span> Operator must reduce Molarity below 2.0 M immediately!
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    if clamped_display_yield < target_yield:
        suggested_m = (target_yield - C_INTERCEPT - (M_TIME * user_time)) / M_MOLARITY
        suggested_m = max(0.1, min(suggested_m, max_molarity))
        st.markdown(f"""
        <div class="autopilot-box">
            <p class="autopilot-text">💡 <strong>Smart Auto-Pilot Suggestion:</strong> To hit your {target_yield}% target efficiently, try setting concentration to <strong>{suggested_m:.2f} M</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- FINANCIAL PROJECTION CARD ---
with st.container(border=True):
    st.markdown("<h3 style='color: #1E293B;'>💼 Economic Feasibility Analysis (100-Ton Pilot Well)</h3>", unsafe_allow_html=True)

    market_price_per_kg = st.slider("📈 Current MREC Market Price (RM/kg) - Risk Analysis", min_value=100.0, max_value=400.0, value=200.0, step=10.0)

    total_ree_kg_in_block = (100 * ree_content) / 1000  
    extracted_ree_kg = total_ree_kg_in_block * (clamped_display_yield / 100)
    revenue_per_block = extracted_ree_kg * market_price_per_kg
    profit_per_block = revenue_per_block - est_opex
    profit_status = "FEASIBLE" if profit_per_block > 0 else "DEFICIT"

    blocks_per_month = 720 / max(user_time, 0.1) 
    monthly_profit = profit_per_block * blocks_per_month

    st.markdown("<br>", unsafe_allow_html=True)
    fin_col1, fin_col2, fin_col3 = st.columns(3)

    if profit_per_block > 0:
        fin_col1.metric("Net Profit (Pilot Well)", f"RM {profit_per_block:,.2f}", f"Gross Revenue: RM {revenue_per_block:,.0f}")
    else:
        fin_col1.metric("Net Profit (Pilot Well)", f"-RM {abs(profit_per_block):,.2f}", f"Deficit: OPEX Exceeds Revenue", delta_color="inverse")

    if monthly_profit > 0:
        fin_col2.metric("Projected Monthly ROI", f"RM {monthly_profit:,.2f}", f"Capacity: {blocks_per_month:.1f} Wells/Month")
        fin_col3.metric("Projected Yearly ROI", f"RM {(monthly_profit*12):,.2f}", "Continuous 24/7 Operation")
    else:
        fin_col2.metric("Projected Monthly ROI", f"-RM {abs(monthly_profit):,.2f}", f"Capacity: {blocks_per_month:.1f} Wells/Month", delta_color="inverse")
        fin_col3.metric("Projected Yearly ROI", f"-RM {abs(monthly_profit*12):,.2f}", "Continuous 24/7 Operation", delta_color="inverse")

    st.markdown("---")
    pie_col, desc_col = st.columns([1.5, 1])
    with pie_col:
        fig_opex = go.Figure(data=[go.Pie(labels=['Base Setup', 'Chemicals (NH4)2SO4', 'Electricity'], values=[base_cost, chemical_cost, electricity_cost], hole=.4, marker_colors=['#94A3B8', '#3B82F6', '#F59E0B'])])
        fig_opex.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=250)
        st.plotly_chart(fig_opex, use_container_width=True)
    with desc_col:
        st.markdown("#### 📊 OPEX Structure")
        st.write("Chemical processing dominates the operational expenditure. Optimization of Molarity directly impacts overall project feasibility.")

# Hubungkan butang PDF di sidebar menggunakan data yang dikira
with export_col2:
    pdf_bytes = generate_pdf_report(user_molarity, user_time, clamped_display_yield, target_yield, profit_per_block, est_opex, profit_status)
    st.download_button(label="📄 PDF REPORT", data=pdf_bytes, file_name="ISL_Executive_Report.pdf", mime="application/pdf", use_container_width=True, type="primary")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 6. GRAPHS, VISUALIZATION & DIGITAL TWIN
# ==========================================
with st.expander("📈 Step 3: Visualizing the Data & Infrastructure", expanded=True):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["⏱️ Time Impact", "🧪 Chemical Impact", "⚖️ Sweet Spot", "🤖 Math Derivation", "🏗️ Digital Twin (SolidWorks)"])

    COLOR_PALETTE = {0.2: '#3498DB', 0.5: '#9B59B6', 1.0: '#E67E22', 1.5: '#2ECC71'}
    CHART_LAYOUT = dict(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=20))

    with tab1:
        fig1 = go.Figure()
        time_line = np.linspace(0, max_time, 8)
        for m in [0.2, 0.5, 1.0, 1.5]:
            fig1.add_trace(go.Scatter(x=time_line, y=[calc_y_mx_c(m, t) for t in time_line], mode='lines+markers', name=f'{m}M', line=dict(width=3, color=COLOR_PALETTE[m])))
        fig1.add_hline(y=target_yield, line_color="#EF4444", line_width=2, line_dash="dash")
        fig1.update_layout(**CHART_LAYOUT, xaxis_title="Hours", yaxis_title="Yield (%)", height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        conc_line = np.linspace(0.1, max_molarity, 8)
        for t_val, col in zip([24, 72, 144], ['#334155', '#3B82F6', '#F59E0B']):
            fig2.add_trace(go.Scatter(x=conc_line, y=[calc_y_mx_c(c_val, t_val) for c_val in conc_line], mode='lines+markers', name=f'{t_val} Hours', line=dict(width=3, color=col)))
        fig2.add_hline(y=target_yield, line_color="#EF4444", line_width=2, line_dash="dash")
        fig2.update_layout(**CHART_LAYOUT, xaxis_title="Molarity (M)", yaxis_title="Yield (%)", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = go.Figure()
        m_range = np.linspace(0.2, max_molarity, 15)
        t_needed = [max(0, min((target_yield - C_INTERCEPT - (M_MOLARITY * m)) / M_TIME, max_time)) for m in m_range]
        fig3.add_trace(go.Scatter(x=t_needed, y=m_range, mode='lines+markers', name='Boundary', line=dict(color='#10B981', width=4)))
        fig3.add_trace(go.Scatter(x=[t_needed[-1]], y=[m_range[-1]], mode='markers', name='Sweet Spot', marker=dict(color='#EF4444', size=16, symbol='star')))
        fig3.update_layout(**CHART_LAYOUT, xaxis_title="Hours", yaxis_title="Molarity (M)", height=400)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.markdown("### 🧮 Model Derivation")
        st.latex(r"Y = 20.0(X_1) + 0.15(X_2) + 12.0")
        st.info("💡 Derived systematically from empirical literature.")
        
    with tab5:
        st.markdown("### 🏗️ Open-Pit Mine Infrastructure (SolidWorks Render)")
        st.write("Integrasi reka bentuk kejuruteraan paip telaga (injection/recovery wells) bagi permodelan lombong terbuka anda.")
        # Tempat letak gambar render SolidWorks anda
        try:
            st.image("cad_mine.png", caption="SolidWorks 3D Render: ISL Well Layout", use_container_width=True)
        except:
            st.info("Sila masukkan fail gambar rekaan SolidWorks anda dengan nama `cad_mine.png` ke dalam folder sistem ini untuk dipaparkan secara automatik.")

# ==========================================
# 7. REFERENCES
# ==========================================
st.markdown("---")
with st.expander("📚 Academic References (APA 7th Edition)"):
    st.markdown("""
    1. **Fendy, N. A., & Ismail, R. (n.d.).** Leaching of NR-REE from IAC using monovalent salt. *Universiti Malaysia Kelantan*.
    2. **Hamka, A. A. M., et al. (2024).** Impacts of ammonium sulfate leaching. *Rudarsko-geološko-naftni zbornik*, 39(1), 27-40.
    3. **He, H., et al. (2023).** Environmental impact of REE development on groundwater. *Water*, 15(2), 263.
    4. **He, Z., et al. (2016).** Process optimization of REE leaching. *Hydrometallurgy*, 164, 1-7.
    5. **Miiro, E. (2023).** *Hydrometallurgical processing of REE from clays*. UCT.
    6. **Moldoveanu, G. A., & Papangelakis, V. G. (2013).** Leaching with ammonium sulphate. *Hydrometallurgy*, 131, 158-166.
    """)
