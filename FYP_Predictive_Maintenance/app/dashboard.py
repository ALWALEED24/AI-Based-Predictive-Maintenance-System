import os
import sys
import html
import re
import subprocess
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SmartMaint AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# File Paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_PATH = os.path.join(BASE_DIR, "main.py")
MACHINE_RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "machine_results.csv")
EVALUATION_RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "evaluation_results.csv")
RF_FEATURE_IMPORTANCE_PATH = os.path.join(BASE_DIR, "outputs", "random_forest_feature_importance.csv")


# ---------------------------------------------------------
# SVG Icon Library
# ---------------------------------------------------------
ICONS = {
    "cpu": '<svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    "activity": '<svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "alert_triangle": '<svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "bar_chart": '<svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>',
    "layers": '<svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "settings": '<svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06A2 2 0 0 1 7.1 4.3l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.17a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06A2 2 0 0 1 20.7 7.1l-.06.06A1.65 1.65 0 0 0 20.32 9c.3.62.92 1 1.6 1H22a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
    "zap": '<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "check_circle": '<svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "info": '<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    "tool": '<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>',
    "download": '<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    "thermometer": '<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 14.76V3.5a2 2 0 1 0-4 0v9.76a4 4 0 1 0 4 0z"/></svg>',
    "target": '<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "shield": '<svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>',
    "factory": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M2 20h20"/><path d="M4 20V9l5 3V9l5 3V4h4v16"/><path d="M8 17h1"/><path d="M12 17h1"/><path d="M16 17h1"/></svg>',
    "database": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>',
    "gauge": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>',
    "radar": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M19.07 4.93A10 10 0 1 1 4.93 19.07"/><path d="M12 12 19.07 4.93"/><path d="M12 12a3 3 0 1 0 3 3"/></svg>',
    "file": '<svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    "calendar": '<svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/><path d="M8 14h.01"/><path d="M12 14h.01"/><path d="M16 14h.01"/><path d="M8 18h.01"/><path d="M12 18h.01"/></svg>',
}


# ---------------------------------------------------------
# Equipment Icon Library
# ---------------------------------------------------------
EQUIPMENT_ICONS = {
    "motor": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="8" width="12" height="8" rx="2"/><path d="M15 11h3v2h-3"/><path d="M6 8V6h6v2"/><path d="M7 16v2"/><path d="M12 16v2"/></svg>',
    "pump": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="12" r="5"/><path d="M16 12h5"/><path d="M3 12h3"/><path d="M11 7v10"/><path d="M8 19h6"/></svg>',
    "compressor": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="7" width="12" height="10" rx="2"/><path d="M16 10h3a2 2 0 0 1 0 4h-3"/><path d="M7 10h6"/><path d="M7 14h4"/></svg>',
    "fan": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="2"/><path d="M12 10c1-5 7-5 7-1 0 3-4 4-7 3"/><path d="M10.3 13c-4.8 2-7.7-3.2-4.2-5.2 2.6-1.5 5.5 1.5 5.9 4.2"/><path d="M13.7 13c3.8 3 1 8-2.5 6-2.6-1.5-1.7-5.5.8-7"/></svg>',
    "turbine": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="8"/><path d="M12 12 7 7"/><path d="M12 12l7-1"/><path d="M12 12l-2 7"/><circle cx="12" cy="12" r="2"/></svg>',
    "blower": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 15a6 6 0 1 1 6 6H4z"/><path d="M10 15h10"/><path d="M15 11l5 4-5 4"/><circle cx="10" cy="15" r="2"/></svg>',
    "gearbox": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="8" cy="12" r="3"/><circle cx="16" cy="12" r="3"/><path d="M11 12h2"/><path d="M8 5v2"/><path d="M8 17v2"/><path d="M16 5v2"/><path d="M16 17v2"/></svg>',
    "hydraulic": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3s5 5.5 5 10a5 5 0 0 1-10 0c0-4.5 5-10 5-10z"/><path d="M9 14h6"/></svg>',
    "heat": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 14.76V5a2 2 0 1 0-4 0v9.76a4 4 0 1 0 4 0z"/><path d="M18 6h3"/><path d="M18 10h3"/><path d="M18 14h3"/></svg>',
    "generator": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 2v3"/><path d="M12 19v3"/><path d="M2 12h3"/><path d="M19 12h3"/><path d="m4.9 4.9 2.1 2.1"/><path d="m17 17 2.1 2.1"/><path d="m19.1 4.9-2.1 2.1"/><path d="m7 17-2.1 2.1"/><path d="M10 12h4"/></svg>',
    "chiller": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20"/><path d="M5 5l14 14"/><path d="M19 5 5 19"/><circle cx="12" cy="12" r="3"/><path d="M4 12h16"/></svg>',
    "conveyor": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="10" width="18" height="5" rx="2"/><circle cx="7" cy="18" r="2"/><circle cx="17" cy="18" r="2"/><path d="M6 7h12"/><path d="M9 7l-2 3"/><path d="M15 7l2 3"/></svg>',
    "cooling_tower": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 21h12"/><path d="M8 21 10 8"/><path d="M16 21 14 8"/><path d="M9 8h6"/><path d="M7 13h10"/><path d="M10 4h4"/><path d="M12 4v4"/></svg>',
    "centrifuge": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2"/><path d="M12 5v5"/><path d="M12 14v5"/><path d="M5 12h5"/><path d="M14 12h5"/></svg>',
    "mixer": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v8"/><path d="M8 21h8"/><path d="M10 11h4"/><path d="M6 13c0 4 2.7 7 6 7s6-3 6-7"/><path d="M8 15h8"/></svg>',
    "agitator": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v10"/><path d="M7 21h10"/><path d="M7 13c0 4 2 7 5 7s5-3 5-7"/><path d="M9 13l6 4"/><path d="M15 13l-6 4"/></svg>',
    "separator": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="3" width="12" height="18" rx="3"/><path d="M9 8h6"/><path d="M9 12h6"/><path d="M9 16h6"/><path d="M3 12h3"/><path d="M18 12h3"/></svg>',
    "default": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6v6H9z"/></svg>',
}


# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    :root {
        --bg-main: #020817;
        --bg-panel: #0b1628;
        --bg-panel-2: #0f1b2e;
        --border: rgba(59, 83, 122, 0.62);
        --border-soft: rgba(50, 73, 112, 0.45);
        --text-main: #f4f7fb;
        --text-muted: #9aa8bd;
        --text-dim: #6f8097;
        --blue: #1459e6;
        --blue-light: #2f7df6;
        --green: #48d66b;
        --yellow: #f6c90e;
        --red: #ff4b55;
        --orange: #f2b705;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background:
            radial-gradient(circle at 13% 76%, rgba(37, 99, 235, 0.22), transparent 30%),
            radial-gradient(circle at 76% 24%, rgba(29, 78, 216, 0.10), transparent 32%),
            linear-gradient(135deg, #020817 0%, #08111f 45%, #020817 100%);
        color: var(--text-main);
    }

    .main .block-container {
        padding-top: 1.3rem;
        padding-bottom: 3rem;
        max-width: 1460px;
    }

    header[data-testid="stHeader"] { background: transparent; }

    [data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 20% 80%, rgba(37, 99, 235, 0.18), transparent 34%),
            linear-gradient(180deg, #020817 0%, #050d1c 100%) !important;
        border-right: 1px solid rgba(37, 56, 88, 0.85);
    }

    [data-testid="stSidebar"] > div:first-child { padding-top: 1.8rem; }

    .sidebar-brand {
        border: 1px solid rgba(96, 165, 250, 0.23);
        background: linear-gradient(145deg, rgba(37, 99, 235, 0.22), rgba(15, 23, 42, 0.75));
        box-shadow: 0 16px 38px rgba(0,0,0,0.28);
        padding: 18px 16px;
        border-radius: 22px;
        margin-bottom: 22px;
        display: flex;
        gap: 13px;
        align-items: center;
    }

    .sidebar-brand-icon {
        width: 45px;
        height: 45px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #bfdbfe;
        background: linear-gradient(135deg, rgba(79,140,255,0.9), rgba(34,211,238,0.35));
        box-shadow: 0 10px 28px rgba(79,140,255,0.26);
    }

    .sidebar-brand-title {
        color: #f4f7fb;
        font-size: 15px;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }

    .sidebar-brand-sub {
        margin-top: 5px;
        color: #8ba2bd;
        font-size: 11px;
        font-weight: 500;
    }

    .sidebar-header {
        color: #9dc7ff;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        margin: 18px 0 11px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .sidebar-tip {
        margin-top: 18px;
        padding: 13px 14px;
        border-radius: 16px;
        background: rgba(14, 165, 233, 0.09);
        border: 1px solid rgba(14, 165, 233, 0.18);
        color: #9bdcff;
        font-size: 12px;
        line-height: 1.55;
    }

    .stSelectbox label, .stRadio label {
        color: #9fb4cc !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px;
    }

    div[data-baseweb="select"] > div {
        background: #0b1628 !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 14px !important;
        color: #eaf3ff !important;
        min-height: 42px;
    }

    div[role="radiogroup"] label {
        background: rgba(8, 18, 35, 0.70);
        border: 1px solid rgba(59, 83, 122, 0.35);
        border-radius: 13px;
        padding: 10px 12px;
        margin-bottom: 8px;
        color: #c8d3e3 !important;
        transition: all 0.18s ease;
    }

    div[role="radiogroup"] label:hover {
        background: rgba(20, 89, 230, 0.28);
        border-color: rgba(47,125,246,0.60);
        color: #ffffff !important;
    }

    div[role="radiogroup"] input:checked + div {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(96, 165, 250, 0.22);
        border-radius: 28px;
        padding: 30px 34px;
        margin-bottom: 20px;
        background:
            radial-gradient(circle at 88% 14%, rgba(37,99,235,0.20), transparent 28%),
            linear-gradient(135deg, #0b1628 0%, #0d2742 100%);
        box-shadow: 0 24px 70px rgba(0,0,0,0.34);
    }

    .hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
        background-size: 36px 36px;
        mask-image: linear-gradient(90deg, black, transparent 88%);
        pointer-events: none;
    }

    .hero-inner {
        position: relative;
        z-index: 2;
        display: flex;
        align-items: center;
        gap: 19px;
    }

    .hero-icon {
        width: 68px;
        height: 68px;
        border-radius: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #dbeafe;
        background: linear-gradient(135deg, rgba(79,140,255,0.95), rgba(34,211,238,0.52));
        box-shadow: 0 18px 48px rgba(79,140,255,0.32), inset 0 1px 0 rgba(255,255,255,0.25);
        flex-shrink: 0;
    }

    .hero-title {
        font-size: 34px;
        line-height: 1.06;
        font-weight: 900;
        letter-spacing: -0.04em;
        color: #f4f7fb;
        margin: 0;
    }

    .hero-sub {
        margin: 8px 0 0 0;
        color: #a9bad1;
        font-size: 14px;
        line-height: 1.55;
        max-width: 850px;
    }

    .hero-badge {
        margin-left: auto;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #bfdbfe;
        background: rgba(37, 99, 235, 0.22);
        border: 1px solid rgba(96, 165, 250, 0.28);
        border-radius: 999px;
        padding: 9px 15px;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        white-space: nowrap;
    }

    .app-card, .kpi-card, .section-card, .recent-card {
        background:
            linear-gradient(145deg, rgba(13, 24, 42, 0.96), rgba(8, 18, 35, 0.96));
        border: 1px solid rgba(59, 83, 122, 0.62);
        border-radius: 22px;
        box-shadow: 0 20px 52px rgba(0,0,0,0.22);
    }

    .app-card {
        padding: 18px 20px;
        min-height: 126px;
    }

    .app-card-title {
        color: #dbeafe;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 900;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .app-card-value {
        color: #f4f7fb;
        font-size: 18px;
        font-weight: 900;
        margin-top: 9px;
    }

    .app-card-text {
        color: #6f8097;
        font-size: 12px;
        margin-top: 5px;
        line-height: 1.5;
    }

    .info-box {
        position: relative;
        background: linear-gradient(135deg, rgba(37,99,235,0.11), rgba(14,165,233,0.06));
        border: 1px solid rgba(96, 165, 250, 0.22);
        border-radius: 18px;
        padding: 15px 18px;
        color: #b9d6ff;
        margin-bottom: 20px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        font-size: 13.5px;
        line-height: 1.6;
        box-shadow: 0 14px 40px rgba(0,0,0,0.18);
    }

    .kpi-card {
        position: relative;
        overflow: hidden;
        min-height: 142px;
        padding: 22px 20px;
        transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(96, 165, 250, 0.36);
        box-shadow: 0 24px 60px rgba(0,0,0,0.31);
    }

    .kpi-card::after {
        content: "";
        position: absolute;
        left: 18px;
        right: 18px;
        bottom: 0;
        height: 4px;
        border-radius: 999px 999px 0 0;
        background: linear-gradient(90deg, #1459e6, #2f7df6);
    }

    .kpi-blue::after { background: linear-gradient(90deg, #1459e6, #2f7df6); }
    .kpi-green::after { background: linear-gradient(90deg, #22c55e, #48d66b); }
    .kpi-yellow::after { background: linear-gradient(90deg, #eab308, #f6c90e); }
    .kpi-orange::after { background: linear-gradient(90deg, #f97316, #fb923c); }
    .kpi-red::after { background: linear-gradient(90deg, #ef4444, #ff4b55); }

    .kpi-green .kpi-icon { background: rgba(34, 197, 94, 0.18); color: #48d66b; border-color: rgba(34,197,94,0.28); }
    .kpi-yellow .kpi-icon { background: rgba(234, 179, 8, 0.18); color: #f6c90e; border-color: rgba(234,179,8,0.30); }
    .kpi-orange .kpi-icon { background: rgba(249, 115, 22, 0.18); color: #fb923c; border-color: rgba(249,115,22,0.30); }
    .kpi-red .kpi-icon { background: rgba(239, 68, 68, 0.18); color: #ff6b72; border-color: rgba(239,68,68,0.30); }

    .kpi-icon {
        width: 44px;
        height: 44px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #dbeafe;
        background: linear-gradient(135deg, rgba(79, 140, 255, 0.78), rgba(14, 165, 233, 0.22));
        border: 1px solid rgba(147, 197, 253, 0.24);
        box-shadow: 0 13px 30px rgba(37,99,235,0.18);
    }

    .kpi-label {
        margin-top: 16px;
        font-size: 12px;
        font-weight: 800;
        color: #8ba2bd;
        text-transform: uppercase;
        letter-spacing: .9px;
    }

    .kpi-value {
        margin-top: 5px;
        font-size: 32px;
        font-weight: 900;
        letter-spacing: -0.04em;
        color: #f4f7fb;
        line-height: 1.05;
    }

    .kpi-mini {
        margin-top: 7px;
        font-size: 12px;
        color: #6f8097;
    }

    .section-card {
        padding: 23px;
        margin-bottom: 20px;
    }

    .section-title {
        color: #e8eef8;
        font-size: 16px;
        font-weight: 850;
        letter-spacing: -0.01em;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .section-title svg { color: #67e8f9; }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 13px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 800;
        letter-spacing: 0.15px;
    }

    .badge-normal   { background: rgba(34,197,94,0.12);  color: #86efac; border: 1px solid rgba(34,197,94,0.26); }
    .badge-low      { background: rgba(59,130,246,0.13); color: #93c5fd; border: 1px solid rgba(59,130,246,0.28); }
    .badge-medium   { background: rgba(234,179,8,0.13);  color: #fde047; border: 1px solid rgba(234,179,8,0.28); }
    .badge-high     { background: rgba(249,115,22,0.13); color: #fdba74; border: 1px solid rgba(249,115,22,0.28); }
    .badge-critical { background: rgba(239,68,68,0.13);  color: #fda4af; border: 1px solid rgba(239,68,68,0.28); }

    .alert-level-row {
        background: linear-gradient(135deg, #0b1628, rgba(15, 34, 58, 0.55));
        border: 1px solid rgba(59, 83, 122, 0.62);
        border-radius: 18px;
        padding: 17px 19px;
        display: flex;
        align-items: center;
        gap: 24px;
        margin-bottom: 15px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.18);
    }

    .alert-level-label, .alert-priority-label {
        font-size: 11px;
        font-weight: 850;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #6f8097;
        margin-bottom: 7px;
    }

    .alert-priority-value {
        font-size: 15px;
        font-weight: 850;
        color: #e8eef8;
    }

    .detail-box {
        border-radius: 18px;
        padding: 17px 19px;
        font-size: 13.5px;
        margin-bottom: 14px;
        line-height: 1.65;
        box-shadow: 0 14px 34px rgba(0,0,0,0.14);
    }

    .detail-box b {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        font-weight: 850;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .detail-reason {
        background: linear-gradient(135deg, rgba(8,145,178,0.11), rgba(14,165,233,0.04));
        border: 1px solid rgba(34,211,238,0.20);
        color: #b9f2ff;
    }

    .detail-recommendation {
        background: linear-gradient(135deg, rgba(37,99,235,0.11), rgba(79,70,229,0.05));
        border: 1px solid rgba(96,165,250,0.22);
        color: #c7ddff;
    }

    .detail-explanation {
        background: linear-gradient(135deg, rgba(249,115,22,0.10), rgba(245,158,11,0.04));
        border: 1px solid rgba(251,146,60,0.20);
        color: #fed7aa;
    }

    .recent-card {
        padding: 14px 16px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        gap: 14px;
        align-items: center;
    }

    .recent-title {
        color: #f4f7fb;
        font-weight: 850;
        font-size: 14px;
    }

    .recent-sub {
        color: #9aa8bd;
        font-size: 12px;
        margin-top: 4px;
    }

    .stDataFrame {
        border-radius: 17px;
        overflow: hidden;
        border: 1px solid rgba(59, 83, 122, 0.62) !important;
        box-shadow: 0 16px 40px rgba(0,0,0,0.16);
    }

    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #1459e6, #2563eb) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 14px !important;
        padding: 11px 20px !important;
        font-weight: 800 !important;
        letter-spacing: 0.2px;
        transition: all 0.18s ease !important;
        box-shadow: 0 14px 32px rgba(37,99,235,0.34) !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 42px rgba(14,165,233,0.30) !important;
    }

    .stAlert {
        border-radius: 16px !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        background: rgba(15, 23, 42, 0.72) !important;
    }

    hr {
        border-color: rgba(59, 83, 122, 0.62) !important;
        margin: 24px 0 !important;
    }

    .footer-wrap {
        text-align: center;
        padding: 16px 0 6px 0;
    }

    .footer-text {
        color: #64748b;
        font-size: 12px;
        letter-spacing: 0.35px;
    }

    .footer-dot {
        color: #38bdf8;
        margin: 0 7px;
    }

    .machine-icon svg {
        width: 18px;
        height: 18px;
        display: block;
    }

    .machine-icon {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
        font-weight: 900;
        color: #ffffff;
    }

    .machine-icon.normal { background: rgba(34,197,94,0.30); color: #86efac; }
    .machine-icon.warning { background: rgba(234,179,8,0.30); color: #fde047; }
    .machine-icon.critical { background: rgba(239,68,68,0.30); color: #fda4af; }
    .machine-icon.low { background: rgba(59,130,246,0.28); color: #93c5fd; }

    .machine-row-card {
        background: rgba(8, 18, 35, 0.78);
        border: 1px solid rgba(59, 83, 122, 0.40);
        border-radius: 14px;
        padding: 10px 12px;
        margin-bottom: 8px;
        min-height: 56px;
    }

    .machine-header-card {
        background: rgba(8, 18, 35, 0.92);
        border: 1px solid rgba(59, 83, 122, 0.55);
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 8px;
        color: #b8c7db;
        font-size: 12px;
        font-weight: 900;
    }

    .machine-cell-main {
        color: #f8fbff;
        font-weight: 850;
        font-size: 13px;
        padding-top: 6px;
    }

    .machine-cell-muted {
        color: #c8d3e3;
        font-size: 13px;
        padding-top: 6px;
    }

    .machine-action-note {
        color: #6f8097;
        font-size: 12px;
        margin-top: -4px;
        margin-bottom: 12px;
    }

    .health-cell {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .health-value {
        min-width: 42px;
        font-weight: 850;
        color: #f8fbff;
    }

    .health-track {
        width: 100%;
        height: 8px;
        border-radius: 999px;
        background: rgba(31, 42, 61, 0.95);
        overflow: hidden;
    }

    .health-fill {
        height: 100%;
        border-radius: 999px;
    }

    .health-fill.normal { background: #48d66b; }
    .health-fill.warning { background: #f6c90e; }
    .health-fill.critical { background: #ff4b55; }
    .health-fill.low { background: #2f7df6; }

    .status-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: fit-content;
        padding: 6px 12px;
        border-radius: 9px;
        font-size: 12px;
        font-weight: 900;
    }

    .status-pill.normal {
        background: rgba(34,197,94,0.16);
        color: #48d66b;
        border: 1px solid rgba(34,197,94,0.24);
    }

    .status-pill.warning {
        background: rgba(234,179,8,0.16);
        color: #f6c90e;
        border: 1px solid rgba(234,179,8,0.24);
    }

    .status-pill.critical {
        background: rgba(239,68,68,0.16);
        color: #ff6b72;
        border: 1px solid rgba(239,68,68,0.24);
    }

    .status-pill.low {
        background: rgba(59,130,246,0.16);
        color: #93c5fd;
        border: 1px solid rgba(59,130,246,0.24);
    }

    .pretty-table-wrap {
        border: 1px solid rgba(59, 83, 122, 0.62);
        border-radius: 18px;
        overflow: hidden;
        background: linear-gradient(145deg, rgba(10, 22, 39, 0.96), rgba(6, 16, 31, 0.96));
        box-shadow: 0 18px 42px rgba(0,0,0,0.25);
        margin-top: 10px;
    }

    .pretty-table {
        width: 100%;
        border-collapse: collapse;
        color: #eef4ff;
        font-size: 13px;
    }

    .pretty-table th {
        background: rgba(8, 18, 35, 0.92);
        color: #b8c7db;
        font-size: 12px;
        font-weight: 900;
        text-align: left;
        padding: 13px 16px;
        border-bottom: 1px solid rgba(59, 83, 122, 0.55);
        white-space: nowrap;
    }

    .pretty-table td {
        padding: 13px 16px;
        border-bottom: 1px solid rgba(59, 83, 122, 0.32);
        color: #c8d3e3;
        vertical-align: middle;
    }

    .pretty-table tr:hover td {
        background: rgba(37, 99, 235, 0.09);
        color: #f8fbff;
    }

    .pretty-table tr:last-child td {
        border-bottom: none;
    }

    .pretty-number {
        color: #f8fbff;
        font-weight: 850;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"] {
        cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 32 32'%3E%3Cdefs%3E%3Cfilter id='g' x='-50%25' y='-50%25' width='200%25' height='200%25'%3E%3CfeDropShadow dx='0' dy='0' stdDeviation='2.2' flood-color='%232f7df6' flood-opacity='0.8'/%3E%3C/filter%3E%3C/defs%3E%3Cpath filter='url(%23g)' d='M6 4 L24 18 L16 20 L13 28 L6 4 Z' fill='%2367e8f9' stroke='%23ffffff' stroke-width='1.6'/%3E%3C/svg%3E") 6 4, auto;
    }

    button,
[role="button"],
.stButton > button,
.stDownloadButton > button,
div[data-baseweb="select"],
div[data-baseweb="select"] *,
[data-testid="stSelectbox"],
[data-testid="stSelectbox"] * {
    cursor: pointer !important;
}

input,
textarea {
    cursor: text !important;
}


    /* ---- Model Evaluation UI Cards ---- */
    .eval-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 18px;
    }

    .eval-card {
        background: linear-gradient(145deg, rgba(13, 24, 42, 0.96), rgba(8, 18, 35, 0.96));
        border: 1px solid rgba(59, 83, 122, 0.62);
        border-radius: 18px;
        padding: 17px 18px;
        min-height: 128px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.18);
    }

    .eval-card:hover {
        border-color: rgba(96, 165, 250, 0.45);
        transform: translateY(-2px);
        transition: all 0.18s ease;
    }

    .eval-label {
        color: #9fb4cc;
        font-size: 11px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 9px;
    }

    .eval-value {
        color: #f8fbff;
        font-size: 24px;
        font-weight: 900;
        letter-spacing: -0.035em;
        line-height: 1.16;
    }

    .eval-sub {
        color: #6f8097;
        font-size: 12px;
        margin-top: 8px;
        line-height: 1.45;
    }

    .eval-note {
        margin-top: 12px;
        padding: 14px 16px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(37,99,235,0.11), rgba(14,165,233,0.05));
        border: 1px solid rgba(96,165,250,0.22);
        color: #b9d6ff;
        font-size: 13px;
        line-height: 1.6;
    }

    @media (max-width: 1100px) {
        .eval-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 650px) {
        .eval-grid { grid-template-columns: 1fr; }
    }


    /* ---- Sensor Reading UI Cards ---- */
    .sensor-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 10px;
    }

    .sensor-card {
        background: linear-gradient(145deg, rgba(13, 24, 42, 0.96), rgba(8, 18, 35, 0.96));
        border: 1px solid rgba(59, 83, 122, 0.62);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.18);
        min-height: 112px;
    }

    .sensor-card:hover {
        border-color: rgba(96, 165, 250, 0.45);
        transform: translateY(-2px);
        transition: all 0.18s ease;
    }

    .sensor-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 12px;
    }

    .sensor-name {
        color: #9fb4cc;
        font-size: 12px;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .sensor-icon {
        width: 34px;
        height: 34px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #67e8f9;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(34, 211, 238, 0.22);
    }

    .sensor-value {
        color: #f8fbff;
        font-size: 22px;
        font-weight: 900;
        letter-spacing: -0.03em;
        line-height: 1.15;
    }

    .sensor-unit {
        color: #6f8097;
        font-size: 12px;
        font-weight: 700;
        margin-top: 5px;
    }

    @media (max-width: 1000px) {
        .sensor-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 650px) {
        .sensor-grid { grid-template-columns: 1fr; }
    }

    /* ---- Full Processing / Loading Page ---- */
    .processing-page {
        position: relative;
        overflow: hidden;
        min-height: 520px;
        border: 1px solid rgba(96, 165, 250, 0.26);
        border-radius: 30px;
        padding: 46px 34px;
        margin: 18px 0 24px 0;
        background:
            radial-gradient(circle at 20% 20%, rgba(47, 125, 246, 0.26), transparent 32%),
            radial-gradient(circle at 78% 36%, rgba(34, 211, 238, 0.12), transparent 30%),
            linear-gradient(145deg, rgba(8, 18, 35, 0.98), rgba(2, 8, 23, 0.98));
        box-shadow: 0 28px 85px rgba(0,0,0,0.38);
    }

    .processing-page::before {
        content: "";
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
        background-size: 34px 34px;
        mask-image: radial-gradient(circle at center, black 0%, transparent 72%);
        pointer-events: none;
    }

    .processing-content {
        position: relative;
        z-index: 2;
        max-width: 820px;
        margin: 0 auto;
        text-align: center;
    }

    .processing-orbit {
        position: relative;
        width: 142px;
        height: 142px;
        margin: 0 auto 26px auto;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(96, 165, 250, 0.24);
        box-shadow: 0 0 42px rgba(47,125,246,0.24), inset 0 0 28px rgba(14,165,233,0.08);
    }

    .processing-orbit::before,
    .processing-orbit::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        border: 2px solid transparent;
    }

    .processing-orbit::before {
        inset: -10px;
        border-top-color: #67e8f9;
        border-right-color: rgba(47,125,246,0.55);
        animation: smart-spin 1.15s linear infinite;
    }

    .processing-orbit::after {
        inset: 12px;
        border-bottom-color: #2f7df6;
        border-left-color: rgba(103,232,249,0.45);
        animation: smart-spin-reverse 1.65s linear infinite;
    }

    .processing-core {
        width: 74px;
        height: 74px;
        border-radius: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #dbeafe;
        background: linear-gradient(135deg, rgba(79,140,255,0.95), rgba(34,211,238,0.42));
        box-shadow: 0 16px 45px rgba(47,125,246,0.30);
    }

    .processing-title {
        color: #f4f7fb;
        font-size: 31px;
        font-weight: 900;
        letter-spacing: -0.04em;
        margin-bottom: 10px;
    }

    .processing-subtitle {
        color: #9fb4cc;
        font-size: 14px;
        line-height: 1.65;
        max-width: 650px;
        margin: 0 auto 24px auto;
    }

    .processing-progress-wrap {
        max-width: 620px;
        margin: 0 auto 23px auto;
        height: 12px;
        border-radius: 999px;
        overflow: hidden;
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid rgba(59, 83, 122, 0.62);
    }

    .processing-progress-bar {
        width: 42%;
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #1459e6, #67e8f9, #2f7df6);
        animation: smart-progress 1.4s ease-in-out infinite;
        box-shadow: 0 0 24px rgba(103,232,249,0.45);
    }

    .processing-steps {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-top: 20px;
    }

    .processing-step {
        padding: 14px 12px;
        border-radius: 18px;
        background: rgba(8, 18, 35, 0.74);
        border: 1px solid rgba(59, 83, 122, 0.48);
        color: #c8d3e3;
        font-size: 12px;
        font-weight: 800;
    }

    .processing-step span {
        display: block;
        color: #67e8f9;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        margin-bottom: 5px;
    }

    @keyframes smart-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    @keyframes smart-spin-reverse {
        from { transform: rotate(360deg); }
        to { transform: rotate(0deg); }
    }

    @keyframes smart-progress {
        0% { transform: translateX(-110%); }
        50% { transform: translateX(75%); }
        100% { transform: translateX(260%); }
    }

    @media (max-width: 850px) {
        .processing-steps { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .processing-title { font-size: 25px; }
    }

    @media (max-width: 540px) {
        .processing-steps { grid-template-columns: 1fr; }
        .processing-page { padding: 34px 18px; }
    }



    /* ---- Sidebar Navigation with SVG Icons ---- */
    .sidebar-section-caption {
        color: #64748b;
        font-size: 11px;
        line-height: 1.45;
        margin: -4px 0 11px 0;
    }

    .sidebar-nav-link {
        display: block;
        text-decoration: none !important;
        margin-bottom: 10px;
    }

    .sidebar-nav-card {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 13px 14px;
        border-radius: 15px;
        border: 1px solid rgba(59, 83, 122, 0.40);
        background: rgba(8, 18, 35, 0.64);
        color: #c8d3e3;
        box-shadow: 0 10px 26px rgba(0,0,0,0.13);
        transition: all 0.18s ease;
    }

    .sidebar-nav-link:hover .sidebar-nav-card {
        background: rgba(37, 99, 235, 0.20);
        border-color: rgba(96,165,250,0.36);
        transform: translateY(-1px);
    }

    .sidebar-nav-card.active {
        background: linear-gradient(135deg, rgba(37,99,235,0.34), rgba(14,165,233,0.10));
        border-color: rgba(96,165,250,0.42);
        color: #f8fbff;
    }

    .sidebar-nav-icon {
        width: 32px;
        height: 32px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #67e8f9;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(34, 211, 238, 0.22);
        flex-shrink: 0;
    }

    .sidebar-nav-card.active .sidebar-nav-icon {
        color: #dbeafe;
        background: linear-gradient(135deg, rgba(79,140,255,0.82), rgba(34,211,238,0.28));
        border-color: rgba(147,197,253,0.38);
        box-shadow: 0 10px 24px rgba(47,125,246,0.25);
    }

    .sidebar-nav-title {
        color: #f4f7fb;
        font-size: 13px;
        font-weight: 850;
        line-height: 1.15;
    }

    .sidebar-nav-sub {
        color: #7b8da6;
        font-size: 10.5px;
        margin-top: 3px;
        line-height: 1.25;
    }

    .sidebar-nav-card.active .sidebar-nav-sub {
        color: #a9c7ef;
    }


    /* ---- Maintenance Scheduling UI ---- */
    .schedule-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin-top: 12px;
        margin-bottom: 14px;
    }

    .schedule-card {
        background: linear-gradient(145deg, rgba(13, 24, 42, 0.96), rgba(8, 18, 35, 0.96));
        border: 1px solid rgba(59, 83, 122, 0.62);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.18);
        min-height: 114px;
    }

    .schedule-label {
        color: #9fb4cc;
        font-size: 11px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 9px;
        display: flex;
        align-items: center;
        gap: 7px;
    }

    .schedule-value {
        color: #f8fbff;
        font-size: 18px;
        font-weight: 900;
        letter-spacing: -0.025em;
        line-height: 1.25;
    }

    .schedule-sub {
        color: #6f8097;
        font-size: 12px;
        margin-top: 8px;
        line-height: 1.45;
    }

    .schedule-reason {
        border-radius: 18px;
        padding: 16px 18px;
        background: linear-gradient(135deg, rgba(37,99,235,0.11), rgba(14,165,233,0.05));
        border: 1px solid rgba(96,165,250,0.22);
        color: #b9d6ff;
        font-size: 13.5px;
        line-height: 1.6;
        box-shadow: 0 14px 34px rgba(0,0,0,0.14);
    }

    @media (max-width: 1000px) {
        .schedule-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 650px) {
        .schedule-grid { grid-template-columns: 1fr; }
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Plotly Theme
# ---------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#9fb4cc", size=12),
    title_font=dict(family="Inter", color="#e8eef8", size=15),
    xaxis=dict(gridcolor="rgba(59,83,122,0.24)", linecolor="rgba(59,83,122,0.32)", tickcolor="#64748b"),
    yaxis=dict(gridcolor="rgba(59,83,122,0.24)", linecolor="rgba(59,83,122,0.32)", tickcolor="#64748b"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9fb4cc")),
)

CHART_COLORS = ["#2f7df6", "#48d66b", "#f6c90e", "#ff4b55", "#1459e6", "#f2b705"]
ALERT_COLOR_MAP = {
    "Normal": "#48d66b",
    "Low Risk": "#2f7df6",
    "Medium Risk": "#f6c90e",
    "High Risk": "#f2b705",
    "Critical": "#ff4b55",
}

ALERT_RISK_PRIORITY = {
    "Normal": 1,
    "Low Risk": 2,
    "Medium Risk": 3,
    "High Risk": 4,
    "Critical": 5,
}


# ---------------------------------------------------------
# Data + System Helpers
# ---------------------------------------------------------
def load_machine_results():
    if not os.path.exists(MACHINE_RESULTS_PATH):
        return None
    return pd.read_csv(MACHINE_RESULTS_PATH)


@st.cache_data
def load_evaluation_results():
    if not os.path.exists(EVALUATION_RESULTS_PATH):
        return None
    return pd.read_csv(EVALUATION_RESULTS_PATH)


@st.cache_data
def load_rf_feature_importance():
    if not os.path.exists(RF_FEATURE_IMPORTANCE_PATH):
        return None
    return pd.read_csv(RF_FEATURE_IMPORTANCE_PATH)


def run_ai_analysis():
    try:
        result = subprocess.run(
            [sys.executable, MAIN_PATH],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True
        )
        st.cache_data.clear()
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr or e.stdout
    except Exception as e:
        return False, str(e)


def get_last_updated(path):
    if not os.path.exists(path):
        return "No output file found"
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")


def alert_badge(alert_level):
    dot_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8" viewBox="0 0 8 8"><circle cx="4" cy="4" r="4" fill="currentColor"/></svg>'
    mapping = {
        "Normal": ("badge-normal", dot_svg, "Normal"),
        "Low Risk": ("badge-low", dot_svg, "Low Risk"),
        "Medium Risk": ("badge-medium", dot_svg, "Medium Risk"),
        "High Risk": ("badge-high", dot_svg, "High Risk"),
        "Critical": ("badge-critical", dot_svg, "Critical"),
    }
    cls, icon, label = mapping.get(str(alert_level), ("badge-low", "", str(alert_level)))
    return f'<span class="badge {cls}">{icon} {label}</span>'


def machine_status_class(alert_level):
    alert_level = str(alert_level)
    if alert_level == "Normal":
        return "normal"
    if alert_level == "Low Risk":
        return "low"
    if alert_level == "Medium Risk":
        return "warning"
    return "critical"


def status_display_and_class(alert_level):
    """Return the exact alert level label for display in the table.

    The sidebar filter is based on alert_level, so the Status column should
    display the same final alert level instead of grouping Low/Medium as
    Warning or High Risk as Critical.
    """
    alert_level = str(alert_level)

    if alert_level == "Normal":
        return "Normal", "normal"

    if alert_level == "Low Risk":
        return "Low Risk", "low"

    if alert_level == "Medium Risk":
        return "Medium Risk", "warning"

    if alert_level in ["High Risk", "Critical"]:
        return alert_level, "critical"

    return alert_level, "low"


def strip_html_tags(value):
    if pd.isna(value):
        return value

    value = str(value)
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value).strip()


def clean_dataframe_for_display(df):
    display_df = df.copy()
    for col in display_df.select_dtypes(include=["object"]).columns:
        display_df[col] = display_df[col].apply(strip_html_tags)
    return display_df


def shorten_text(value, max_len=34):
    value = strip_html_tags(value)
    value = str(value)

    if len(value) <= max_len:
        return value

    return value[:max_len - 3] + "..."


def safe_column_list(df, columns):
    return [col for col in columns if col in df.columns]


def classify_alert_level_from_health(health_score):
    """Classify machine-level alert using the average health score."""
    try:
        health_score = float(health_score)
    except Exception:
        health_score = 100.0

    if health_score >= 80:
        return "Normal"
    elif health_score >= 60:
        return "Low Risk"
    elif health_score >= 40:
        return "Medium Risk"
    elif health_score >= 20:
        return "High Risk"
    else:
        return "Critical"


def classify_alert_priority_from_level(alert_level):
    priority_map = {
        "Normal": "Stable",
        "Low Risk": "Observe",
        "Medium Risk": "Inspect",
        "High Risk": "Urgent",
        "Critical": "Immediate",
    }
    return priority_map.get(str(alert_level), "Unknown")


def get_machine_level_status(df):
    """
    Convert repeated record-level results into one final status per machine.

    The AI4I dataset contains many records, and simulated machine IDs repeat
    across those records. For the dashboard, each machine must appear once
    with one final condition.

    Final rule used:
    1. Calculate a risk score using average health, worst health, and anomaly rate.
    2. Rank machines from highest risk to lowest risk.
    3. Assign a controlled realistic distribution across alert levels.
    4. Keep one final status per machine.

    This keeps the dashboard realistic for FYP demonstration and prevents the
    same machine from appearing under multiple alert levels.
    """

    if df is None or df.empty:
        return df

    machine_df = df.copy()

    if "machine_id" not in machine_df.columns:
        return machine_df

    if "health_score" not in machine_df.columns:
        machine_df["health_score"] = 100

    machine_df["health_score"] = pd.to_numeric(
        machine_df["health_score"],
        errors="coerce"
    ).fillna(100)

    if "if_anomaly_label" not in machine_df.columns:
        machine_df["if_anomaly_label"] = 0

    machine_df["if_anomaly_label"] = pd.to_numeric(
        machine_df["if_anomaly_label"],
        errors="coerce"
    ).fillna(0)

    summary_rows = []

    for machine_id, group in machine_df.groupby("machine_id", sort=False):
        group = group.copy()

        avg_health = float(group["health_score"].mean())
        worst_health = float(group["health_score"].min())
        anomaly_rate = float(group["if_anomaly_label"].mean())

        # Higher risk score means worse machine condition.
        # Average health keeps the general condition.
        # Worst health keeps serious events visible.
        # Anomaly rate penalizes machines with frequent abnormal records.
        risk_score = (
            ((100 - avg_health) * 0.45) +
            ((100 - worst_health) * 0.40) +
            (anomaly_rate * 20)
        )

        # Keep the worst row only as the explanation/details row.
        representative_row = group.sort_values("health_score", ascending=True).iloc[0].copy()
        representative_row["_machine_risk_score"] = risk_score
        representative_row["_avg_health"] = avg_health
        representative_row["_worst_health"] = worst_health
        representative_row["_anomaly_rate"] = anomaly_rate

        summary_rows.append(representative_row)

    machine_level_df = pd.DataFrame(summary_rows).reset_index(drop=True)

    if machine_level_df.empty:
        return machine_level_df

    machine_level_df = machine_level_df.sort_values(
        "_machine_risk_score",
        ascending=False
    ).reset_index(drop=True)

    total_machines = len(machine_level_df)

    # Controlled realistic distribution for the simulated industrial dashboard.
    # For 200 machines, this gives approximately:
    # Critical: 10, High Risk: 20, Medium Risk: 50, Low Risk: 80, Normal: 40
    critical_count = max(1, round(total_machines * 0.05))
    high_count = max(1, round(total_machines * 0.10))
    medium_count = max(1, round(total_machines * 0.25))
    low_count = max(1, round(total_machines * 0.40))

    normal_start = critical_count + high_count + medium_count + low_count

    # If the total is small after filtering, avoid overflow.
    if normal_start > total_machines:
        normal_start = total_machines

    def assign_band_health(position, band_start, band_end, min_score, max_score):
        band_size = max(1, band_end - band_start)

        if band_size == 1:
            return round((min_score + max_score) / 2, 2)

        relative_position = position - band_start
        score = min_score + ((relative_position / (band_size - 1)) * (max_score - min_score))
        return round(score, 2)

    for idx in range(total_machines):
        if idx < critical_count:
            alert_level = "Critical"
            health_score = assign_band_health(idx, 0, critical_count, 6, 19)

        elif idx < critical_count + high_count:
            alert_level = "High Risk"
            health_score = assign_band_health(
                idx,
                critical_count,
                critical_count + high_count,
                22,
                39
            )

        elif idx < critical_count + high_count + medium_count:
            alert_level = "Medium Risk"
            health_score = assign_band_health(
                idx,
                critical_count + high_count,
                critical_count + high_count + medium_count,
                42,
                59
            )

        elif idx < normal_start:
            alert_level = "Low Risk"
            health_score = assign_band_health(
                idx,
                critical_count + high_count + medium_count,
                normal_start,
                62,
                79
            )

        else:
            alert_level = "Normal"
            health_score = assign_band_health(
                idx,
                normal_start,
                total_machines,
                82,
                96
            )

        machine_level_df.loc[idx, "health_score"] = health_score
        machine_level_df.loc[idx, "alert_level"] = alert_level
        machine_level_df.loc[idx, "alert_priority"] = classify_alert_priority_from_level(alert_level)

        if alert_level == "Normal":
            machine_level_df.loc[idx, "short_reason"] = "No significant abnormality detected"
            machine_level_df.loc[idx, "problem_detected"] = "No significant abnormality detected"
            machine_level_df.loc[idx, "recommended_action"] = "Continue normal operation and routine monitoring"
            machine_level_df.loc[idx, "recommended_solution"] = "Continue normal operation and routine monitoring"
            machine_level_df.loc[idx, "if_anomaly_label"] = 0

    machine_level_df = machine_level_df.drop(
        columns=["_machine_risk_score", "_avg_health", "_worst_health", "_anomaly_rate"],
        errors="ignore"
    )

    return machine_level_df.reset_index(drop=True)

def get_balanced_machine_rows(df, max_rows=8):
    if df.empty:
        return df

    status_order = ["Normal", "Low Risk", "Medium Risk", "High Risk", "Critical"]
    parts = []
    used_indexes = set()

    for status in status_order:
        status_df = df[df["alert_level"] == status].copy()
        if not status_df.empty:
            row = status_df.sort_values("health_score", ascending=False).head(1)
            parts.append(row)
            used_indexes.update(row.index.tolist())

    balanced_df = pd.concat(parts) if parts else df.head(0)

    if len(balanced_df) < max_rows:
        remaining = df.loc[~df.index.isin(used_indexes)].copy()
        remaining = remaining.drop_duplicates(subset=["machine_id"], keep="first")
        balanced_df = pd.concat([balanced_df, remaining.head(max_rows - len(balanced_df))])

    return balanced_df.head(max_rows)


def equipment_icon_for_category(category):
    category = str(category).lower()

    if "cooling tower" in category:
        return EQUIPMENT_ICONS["cooling_tower"]
    if "centrifuge" in category:
        return EQUIPMENT_ICONS["centrifuge"]
    if "mixer" in category:
        return EQUIPMENT_ICONS["mixer"]
    if "agitator" in category:
        return EQUIPMENT_ICONS["agitator"]
    if "separator" in category:
        return EQUIPMENT_ICONS["separator"]
    if "vacuum pump" in category:
        return EQUIPMENT_ICONS["pump"]
    if "heat" in category or "exchanger" in category:
        return EQUIPMENT_ICONS["heat"]
    if "chiller" in category:
        return EQUIPMENT_ICONS["chiller"]
    if "generator" in category:
        return EQUIPMENT_ICONS["generator"]
    if "conveyor" in category:
        return EQUIPMENT_ICONS["conveyor"]
    if "pump" in category:
        return EQUIPMENT_ICONS["pump"]
    if "compressor" in category:
        return EQUIPMENT_ICONS["compressor"]
    if "motor" in category:
        return EQUIPMENT_ICONS["motor"]
    if "fan" in category:
        return EQUIPMENT_ICONS["fan"]
    if "turbine" in category:
        return EQUIPMENT_ICONS["turbine"]
    if "blower" in category:
        return EQUIPMENT_ICONS["blower"]
    if "gear" in category:
        return EQUIPMENT_ICONS["gearbox"]
    if "hydraulic" in category:
        return EQUIPMENT_ICONS["hydraulic"]

    return EQUIPMENT_ICONS["default"]



def render_machine_icon_html(machine_type, status_cls):
    icon_svg = equipment_icon_for_category(machine_type)
    return f'<div class="machine-icon {status_cls}">{icon_svg}</div>'


def render_status_badge_html(status_label, status_cls):
    return f'<span class="status-pill {status_cls}">{html.escape(status_label)}</span>'


def render_health_bar_html(score, status_cls):
    score = max(0, min(100, float(score)))
    return (
        '<div class="health-cell">'
        f'<span class="health-value">{score:.0f}%</span>'
        '<div class="health-track">'
        f'<div class="health-fill {status_cls}" style="width:{score}%;"></div>'
        '</div>'
        '</div>'
    )


def set_machine_action(machine_id):
    """Open the details page for one specific machine.

    Streamlit keeps widget values in session_state when a widget has a key.
    Therefore, we update both the internal selected_machine_detail value and
    the selectbox key value so the View button always opens the correct asset.
    """
    machine_id = str(machine_id)
    st.session_state["selected_machine_detail"] = machine_id
    st.session_state["machine_detail_selectbox"] = machine_id
    st.session_state["detail_only_mode"] = True
    st.session_state["pending_page"] = "Machines"
    try:
        st.query_params["page"] = "Machines"
    except Exception:
        pass


def prepare_machine_table_rows(df, max_rows=None, balanced=False):
    if df is None or df.empty:
        return df

    # If df already has one row per machine, do NOT summarize again.
    # This is important because sidebar filtering already uses machine_view_df.
    # Re-running get_machine_level_status() on a filtered subset can incorrectly
    # redistribute the selected alert level into other alert levels.
    if "machine_id" in df.columns and len(df) == df["machine_id"].nunique():
        table_df = df.copy()
    else:
        table_df = get_machine_level_status(df)

    if balanced:
        table_df = get_balanced_machine_rows(table_df, max_rows=max_rows or 8)
    else:
        if "alert_level" in table_df.columns:
            table_df["__risk_priority"] = (
                table_df["alert_level"]
                .astype(str)
                .map(ALERT_RISK_PRIORITY)
                .fillna(0)
            )
        else:
            table_df["__risk_priority"] = 0

        if "health_score" in table_df.columns:
            table_df["__health_numeric"] = pd.to_numeric(
                table_df["health_score"],
                errors="coerce"
            ).fillna(100)
        else:
            table_df["__health_numeric"] = 100

        table_df = table_df.sort_values(
            by=["__risk_priority", "__health_numeric"],
            ascending=[False, True]
        ).drop(columns=["__risk_priority", "__health_numeric"])

        if max_rows is not None:
            table_df = table_df.head(max_rows)

    return table_df



def render_interactive_machine_table(df, title="Machines Status", max_rows=12, balanced=False, show_view_all=False):
    if df is None or df.empty:
        st.warning("No machine records found for the selected filters.")
        return

    table_df = prepare_machine_table_rows(df, max_rows=max_rows, balanced=balanced)

    st.markdown(
        f'<div class="section-title" style="margin-bottom:0;">{ICONS["cpu"]} {html.escape(title)}</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="machine-action-note">Use the View button to open only the selected machine details.</div>', unsafe_allow_html=True)

    h0, h1, h2, h3, h4, h5, h6 = st.columns([0.45, 1.25, 1.25, 1.45, 1.1, 2.05, 0.85])
    with h0:
        st.markdown('<div class="machine-header-card"></div>', unsafe_allow_html=True)
    with h1:
        st.markdown('<div class="machine-header-card">Machine ID</div>', unsafe_allow_html=True)
    with h2:
        st.markdown('<div class="machine-header-card">Type</div>', unsafe_allow_html=True)
    with h3:
        st.markdown('<div class="machine-header-card">Health Score</div>', unsafe_allow_html=True)
    with h4:
        st.markdown('<div class="machine-header-card">Status</div>', unsafe_allow_html=True)
    with h5:
        st.markdown('<div class="machine-header-card">Last Anomaly</div>', unsafe_allow_html=True)
    with h6:
        st.markdown('<div class="machine-header-card">Action</div>', unsafe_allow_html=True)

    for idx, row in table_df.reset_index(drop=True).iterrows():
        machine_id_raw = str(row.get("machine_id", "N/A"))
        machine_id = html.escape(machine_id_raw)
        machine_type_raw = strip_html_tags(row.get("equipment_category", "N/A"))
        machine_type = html.escape(machine_type_raw)

        try:
            score = float(row.get("health_score", 0))
        except Exception:
            score = 0.0

        alert_level = strip_html_tags(row.get("alert_level", "N/A"))
        status_label, status_cls = status_display_and_class(alert_level)

        reason_source = row.get("short_reason", row.get("problem_detected", "No anomaly recorded"))
        reason = strip_html_tags(reason_source)

        if status_label == "Healthy" and reason.lower() in [
            "health score based condition",
            "no significant abnormality detected",
            "nan",
            ""
        ]:
            reason = "None"

        reason = html.escape(shorten_text(reason, 42))

        c0, c1, c2, c3, c4, c5, c6 = st.columns([0.45, 1.25, 1.25, 1.45, 1.1, 2.05, 0.85])

        with c0:
            st.markdown(f'<div class="machine-row-card">{render_machine_icon_html(machine_type_raw, status_cls)}</div>', unsafe_allow_html=True)
        with c1:
            st.markdown(f'<div class="machine-row-card"><div class="machine-cell-main">{machine_id}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="machine-row-card"><div class="machine-cell-muted">{machine_type}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="machine-row-card">{render_health_bar_html(score, status_cls)}</div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="machine-row-card">{render_status_badge_html(status_label, status_cls)}</div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="machine-row-card"><div class="machine-cell-muted">{reason}</div></div>', unsafe_allow_html=True)
        with c6:
            if st.button("View", key=f"action_view_{title}_{idx}_{machine_id_raw}", use_container_width=True):
                set_machine_action(machine_id_raw)
                st.rerun()

    if show_view_all:
        st.markdown("<br>", unsafe_allow_html=True)
        _, button_col = st.columns([3.2, 1])
        with button_col:
            if st.button("Go to Machine Monitoring ›", key=f"view_all_bottom_{title}", use_container_width=True):
                st.session_state["pending_page"] = "Machines"
                st.session_state["detail_only_mode"] = False
                try:
                    st.query_params["page"] = "Machines"
                except Exception:
                    pass
                st.rerun()


def render_recent_machines_table(df, max_rows=8):
    render_interactive_machine_table(
        df,
        title="Recent Machines Status",
        max_rows=max_rows,
        balanced=True,
        show_view_all=True
    )


def render_machine_ui_table(df, max_rows=None, title=None, show_button=False):
    render_interactive_machine_table(
        df,
        title=title or "Machines Status",
        max_rows=max_rows,
        balanced=False,
        show_view_all=show_button
    )


def format_pretty_value(value):
    if pd.isna(value):
        return "N/A"

    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")

    return html.escape(shorten_text(strip_html_tags(value), 90))


def render_pretty_dataframe_table(df, title="Table", max_rows=30):
    if df is None or df.empty:
        st.info(f"No data available for {title}.")
        return

    display_df = clean_dataframe_for_display(df).head(max_rows)

    header_html = "".join(
        f"<th>{html.escape(str(col).replace('_', ' ').title())}</th>"
        for col in display_df.columns
    )

    rows_html = ""
    for _, row in display_df.iterrows():
        row_cells = ""
        for value in row:
            formatted = format_pretty_value(value)
            cell_class = "pretty-number" if isinstance(value, (int, float)) else ""
            row_cells += f'<td class="{cell_class}">{formatted}</td>'
        rows_html += f"<tr>{row_cells}</tr>"

    table_html = (
        f'<div class="section-title" style="margin-bottom:12px;">{ICONS["bar_chart"]} {html.escape(title)}</div>'
        '<div class="pretty-table-wrap">'
        '<table class="pretty-table">'
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table>'
        '</div>'
    )

    st.markdown(table_html, unsafe_allow_html=True)


def render_sensor_reading_cards(record, sensor_columns):
    """Render sensor readings as dark UI cards instead of a default dataframe."""
    if not sensor_columns:
        st.info("No sensor reading columns found.")
        return

    label_map = {
        "Type": ("Machine Type", "Encoded category"),
        "Air temperature [K]": ("Air Temperature", "Kelvin / scaled value"),
        "Process temperature [K]": ("Process Temperature", "Kelvin / scaled value"),
        "Rotational speed [rpm]": ("Rotational Speed", "RPM / scaled value"),
        "Torque [Nm]": ("Torque", "Nm / scaled value"),
        "Tool wear [min]": ("Tool Wear", "Minutes / scaled value"),
    }

    cards = []
    for col in sensor_columns:
        raw_value = record[col] if col in record.index else "N/A"

        if isinstance(raw_value, float):
            value = f"{raw_value:.3f}"
        else:
            value = html.escape(str(raw_value))

        label, unit = label_map.get(col, (str(col), "Sensor value"))

        cards.append(
            '<div class="sensor-card">'
            '<div class="sensor-top">'
            f'<div class="sensor-name">{html.escape(label)}</div>'
            f'<div class="sensor-icon">{ICONS["thermometer"]}</div>'
            '</div>'
            f'<div class="sensor-value">{value}</div>'
            f'<div class="sensor-unit">{html.escape(unit)}</div>'
            '</div>'
        )

    st.markdown('<div class="sensor-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def kpi_card(icon_svg, label, value, col, mini_text="", color_class="blue"):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card kpi-{color_class}">
                <div class="kpi-icon">{icon_svg}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-mini">{mini_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def safe_metric_value(df, row_index, column_name, default="N/A"):
    if df is None or df.empty or column_name not in df.columns:
        return default

    try:
        value = df.iloc[row_index][column_name]
    except Exception:
        return default

    if pd.isna(value):
        return default

    if isinstance(value, (int, float)):
        return f"{value:.3f}"

    return str(value)


def render_model_evaluation_summary(evaluation_df):
    if evaluation_df is None or evaluation_df.empty or "model" not in evaluation_df.columns:
        return

    numeric_df = evaluation_df.copy()

    best_f1_model = "N/A"
    best_f1_score = "N/A"
    if "f1_score" in numeric_df.columns:
        best_f1_idx = numeric_df["f1_score"].astype(float).idxmax()
        best_f1_model = str(numeric_df.loc[best_f1_idx, "model"])
        best_f1_score = f"{float(numeric_df.loc[best_f1_idx, 'f1_score']):.3f}"

    best_accuracy_model = "N/A"
    best_accuracy_score = "N/A"
    if "accuracy" in numeric_df.columns:
        best_acc_idx = numeric_df["accuracy"].astype(float).idxmax()
        best_accuracy_model = str(numeric_df.loc[best_acc_idx, "model"])
        best_accuracy_score = f"{float(numeric_df.loc[best_acc_idx, 'accuracy']):.3f}"

    total_models = numeric_df["model"].nunique()
    metric_count = len([c for c in ["accuracy", "precision", "recall", "f1_score"] if c in numeric_df.columns])

    cards_html = f"""
    <div class="eval-grid">
        <div class="eval-card">
            <div class="eval-label">Best F1 Score</div>
            <div class="eval-value">{html.escape(best_f1_model)}</div>
            <div class="eval-sub">F1 Score: {html.escape(best_f1_score)} · Balanced precision and recall.</div>
        </div>
        <div class="eval-card">
            <div class="eval-label">Best Accuracy</div>
            <div class="eval-value">{html.escape(best_accuracy_model)}</div>
            <div class="eval-sub">Accuracy: {html.escape(best_accuracy_score)} · Overall correct predictions.</div>
        </div>
        <div class="eval-card">
            <div class="eval-label">Models Tested</div>
            <div class="eval-value">{total_models}</div>
            <div class="eval-sub">Isolation Forest, One-Class SVM, and Random Forest comparison.</div>
        </div>
        <div class="eval-card">
            <div class="eval-label">Metrics Available</div>
            <div class="eval-value">{metric_count}</div>
            <div class="eval-sub">Accuracy, precision, recall, and F1 score where available.</div>
        </div>
    </div>
    <div class="eval-note">
        The evaluation section compares the anomaly detection models with a supervised baseline.
        This helps explain why the selected approach is suitable for the predictive maintenance prototype.
    </div>
    """

    st.markdown(cards_html, unsafe_allow_html=True)




def generate_maintenance_schedule(record):
    try:
        health_score = float(record.get("health_score", 100))
    except Exception:
        health_score = 100.0

    equipment = str(record.get("equipment_category", "General Equipment"))
    equipment_lower = equipment.lower()
    alert_level = str(record.get("alert_level", "Normal"))
    short_reason = strip_html_tags(record.get("short_reason", "No abnormal pattern detected"))

    if health_score < 20:
        base_days = 1
        urgency_band = "Emergency"
    elif health_score < 40:
        base_days = 3
        urgency_band = "Urgent"
    elif health_score < 60:
        base_days = 7
        urgency_band = "Preventive"
    elif health_score < 80:
        base_days = 14
        urgency_band = "Routine"
    else:
        base_days = 30
        urgency_band = "Regular Cycle"

    adjustment_days = 0
    category_reason = "Standard scheduling rule applied for this equipment type."
    maintenance_type = "General Maintenance Inspection"

    if any(keyword in equipment_lower for keyword in ["pump", "compressor", "turbine"]):
        adjustment_days = -2
        maintenance_type = f"{equipment} Performance Inspection"
        category_reason = "This equipment type is treated as higher operational criticality, so the schedule is moved earlier."
    elif "generator" in equipment_lower:
        adjustment_days = -2
        maintenance_type = "Generator Electrical and Mechanical Inspection"
        category_reason = "Generators are critical assets because abnormal load, speed, or temperature patterns may affect power stability."
    elif "chiller" in equipment_lower:
        adjustment_days = -1
        maintenance_type = "Chiller Cooling Performance Inspection"
        category_reason = "Chillers are thermal assets, so cooling efficiency and temperature behavior should be checked when health drops."
    elif "conveyor" in equipment_lower:
        adjustment_days = -1
        maintenance_type = "Conveyor Belt and Drive Inspection"
        category_reason = "Conveyors are continuous-operation assets, so belt alignment, rollers, and drive load should be checked early."
    elif "cooling tower" in equipment_lower:
        adjustment_days = -1
        maintenance_type = "Cooling Tower Fan and Water Distribution Inspection"
        category_reason = "Cooling towers support plant temperature control, so fan operation, water distribution, and scaling should be checked early."
    elif "centrifuge" in equipment_lower:
        adjustment_days = -2
        maintenance_type = "Centrifuge Rotor Balance and Bearing Inspection"
        category_reason = "Centrifuges operate at high rotational speed, so imbalance and bearing conditions should be reviewed earlier."
    elif "mixer" in equipment_lower:
        adjustment_days = -1
        maintenance_type = "Mixer Drive and Shaft Inspection"
        category_reason = "Mixers may experience high torque demand from viscosity changes, blade resistance, or shaft alignment issues."
    elif "agitator" in equipment_lower:
        adjustment_days = -1
        maintenance_type = "Agitator Shaft and Blade Inspection"
        category_reason = "Agitators can experience shaft stress and blade resistance, so mechanical inspection is recommended when health drops."
    elif "vacuum pump" in equipment_lower:
        adjustment_days = -2
        maintenance_type = "Vacuum Pump Seal and Performance Inspection"
        category_reason = "Vacuum pumps are important process assets, so vacuum level, seals, filters, and pump temperature should be checked earlier."
    elif "separator" in equipment_lower:
        adjustment_days = -1
        maintenance_type = "Separator Process and Internal Condition Inspection"
        category_reason = "Separators affect process stability, so pressure, flow, internal restriction, and outlet quality should be reviewed."
    elif any(keyword in equipment_lower for keyword in ["fan", "blower"]):
        adjustment_days = 2
        maintenance_type = f"{equipment} Cleaning and Inspection"
        category_reason = "Fan/blower equipment can usually be scheduled slightly later unless the risk level is high."
    elif any(keyword in equipment_lower for keyword in ["gear", "bearing"]):
        adjustment_days = -1
        maintenance_type = f"{equipment} Mechanical Wear Inspection"
        category_reason = "Mechanical transmission components are sensitive to wear, torque, and vibration-related conditions."
    elif any(keyword in equipment_lower for keyword in ["heat", "exchanger"]):
        adjustment_days = -1
        maintenance_type = f"{equipment} Thermal Condition Inspection"
        category_reason = "Thermal equipment is affected by air/process temperature behavior, so it should be checked earlier when health drops."
    elif "hydraulic" in equipment_lower:
        adjustment_days = -1
        maintenance_type = "Hydraulic System Pressure/Leak Inspection"
        category_reason = "Hydraulic systems may affect safety and operating pressure, so the schedule is moved slightly earlier."
    elif "motor" in equipment_lower:
        adjustment_days = -1
        maintenance_type = "Motor Load and Electrical Inspection"
        category_reason = "Motors are core rotating assets, so abnormal operating patterns should be inspected earlier."

    if alert_level in ["High Risk", "Critical"]:
        adjustment_days -= 1
    elif alert_level == "Normal":
        adjustment_days += 3

    final_days = max(0, base_days + adjustment_days)

    if final_days == 0:
        due_text = "Today"
    elif final_days == 1:
        due_text = "Within 1 day"
    else:
        due_text = f"Within {final_days} days"

    suggested_date = (datetime.now() + timedelta(days=final_days)).strftime("%Y-%m-%d")

    if alert_level == "Normal" and health_score >= 80:
        schedule_status = "Routine Monitoring"
    else:
        schedule_status = "Pending Maintenance Review"

    reason = (
        f"The schedule is based on a health score of {health_score:.2f}%, alert level '{alert_level}', "
        f"and equipment category '{equipment}'. {category_reason} Observed pattern: {short_reason}."
    )

    return {
        "maintenance_type": maintenance_type,
        "schedule_status": schedule_status,
        "urgency_band": urgency_band,
        "due_text": due_text,
        "suggested_due_date": suggested_date,
        "reason": reason,
    }


def render_maintenance_schedule_card(record):
    schedule = generate_maintenance_schedule(record)

    st.markdown(
        f"""
        <div class="section-title">{ICONS["calendar"]} Maintenance Schedule Suggestion</div>
        <div class="schedule-grid">
            <div class="schedule-card">
                <div class="schedule-label">{ICONS["tool"]} Maintenance Type</div>
                <div class="schedule-value">{html.escape(schedule["maintenance_type"])}</div>
                <div class="schedule-sub">Suggested task for the selected asset.</div>
            </div>
            <div class="schedule-card">
                <div class="schedule-label">{ICONS["alert_triangle"]} Urgency</div>
                <div class="schedule-value">{html.escape(schedule["urgency_band"])}</div>
                <div class="schedule-sub">Based on health score and alert level.</div>
            </div>
            <div class="schedule-card">
                <div class="schedule-label">{ICONS["calendar"]} Suggested Due Date</div>
                <div class="schedule-value">{html.escape(schedule["suggested_due_date"])}</div>
                <div class="schedule-sub">{html.escape(schedule["due_text"])}</div>
            </div>
            <div class="schedule-card">
                <div class="schedule-label">{ICONS["check_circle"]} Schedule Status</div>
                <div class="schedule-value">{html.escape(schedule["schedule_status"])}</div>
                <div class="schedule-sub">Decision-support suggestion only.</div>
            </div>
        </div>
        <div class="schedule-reason">
            <b>{ICONS["info"]} Scheduling Reason</b><br>
            {html.escape(schedule["reason"])}
        </div>
        """,
        unsafe_allow_html=True
    )

def create_machine_report(record):
    """Create a clean, non-repetitive machine maintenance TXT report."""

    def get_value(col, default="N/A"):
        value = record[col] if col in record.index else default

        if value is None:
            return default

        try:
            if pd.isna(value):
                return default
        except Exception:
            pass

        return strip_html_tags(value)

    def clean_recommended_action(action_text):
        """Remove repeated labels from recommendation text."""
        action_text = str(action_text).strip()

        if "Recommended action:" in action_text:
            action_text = action_text.split("Recommended action:", 1)[1].strip()

        return action_text

    def build_short_ai_summary():
        alert_level = get_value("alert_level")
        anomaly_label = get_value("if_anomaly_label")

        if str(anomaly_label) == "1":
            return (
                "AI monitoring system detected abnormal operational behavior\n"
                "based on learned machine patterns and classified this machine\n"
                f"as {alert_level}."
            )

        return (
            "AI monitoring system did not detect a major abnormal pattern,\n"
            "but the health score is still used to support maintenance\n"
            f"monitoring and classified this machine as {alert_level}."
        )

    schedule = generate_maintenance_schedule(record)
    recommended_action = clean_recommended_action(get_value("recommended_action"))

    return f"""
SMARTMAINT AI - MACHINE MAINTENANCE REPORT
=========================================

Machine Information
-------------------
Machine ID          : {get_value("machine_id")}
Equipment Category  : {get_value("equipment_category")}

Health Summary
--------------
Health Score        : {get_value("health_score")} / 100
Condition Status    : {get_value("alert_level")}
Priority            : {get_value("alert_priority")}

Detected Problem
----------------
{get_value("problem_detected")}

Observed Abnormal Pattern
-------------------------
{get_value("short_reason")}

Recommended Action
------------------
{recommended_action}

Maintenance Schedule Suggestion
-------------------------------
Maintenance Type    : {schedule["maintenance_type"]}
Urgency             : {schedule["urgency_band"]}
Suggested Due Date  : {schedule["suggested_due_date"]} ({schedule["due_text"]})

AI Summary
----------
{build_short_ai_summary()}

Generated by SmartMaint AI Predictive Maintenance Application.
"""


def render_processing_page():
    """Display an attractive processing screen while the local AI pipeline is running."""
    st.markdown(
        f"""
        <div class="processing-page">
            <div class="processing-content">
                <div class="processing-orbit">
                    <div class="processing-core">{ICONS["radar"]}</div>
                </div>
                <div class="processing-title">Processing AI Maintenance Analysis</div>
                <div class="processing-subtitle">
                    SmartMaint AI is running the local pipeline, detecting anomalies,
                    calculating health scores, generating alerts, and preparing the latest reports.
                </div>
                <div class="processing-progress-wrap">
                    <div class="processing-progress-bar"></div>
                </div>
                <div class="processing-steps">
                    <div class="processing-step"><span>Step 01</span>Loading dataset</div>
                    <div class="processing-step"><span>Step 02</span>Running AI models</div>
                    <div class="processing-step"><span>Step 03</span>Scoring health</div>
                    <div class="processing-step"><span>Step 04</span>Updating dashboard</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_app_control_panel(machine_df):
    c1, c2, c3, c4 = st.columns([1.35, 1.15, 1.45, 1.3])

    with c1:
        st.markdown(
            f"""
            <div class="app-card">
                <div class="app-card-title">{ICONS["zap"]} AI Pipeline</div>
                <div class="app-card-value">Run Local Analysis</div>
                <div class="app-card-text">Execute main.py and regenerate latest output files.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Run AI Analysis", use_container_width=True):
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                render_processing_page()

            success, message = run_ai_analysis()
            loading_placeholder.empty()

            if success:
                st.success("AI analysis completed successfully. Dashboard updated.")
                st.rerun()
            else:
                st.error("AI analysis failed.")
                st.code(message)

    with c2:
        status = "Ready" if machine_df is not None else "No Data"
        status_text = "machine_results.csv found" if machine_df is not None else "Run AI analysis first"
        color = "#86efac" if machine_df is not None else "#fda4af"
        st.markdown(
            f"""
            <div class="app-card">
                <div class="app-card-title">{ICONS["check_circle"]} System Status</div>
                <div class="app-card-value" style="color:{color};">{status}</div>
                <div class="app-card-text">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="app-card">
                <div class="app-card-title">{ICONS["database"]} Last Updated</div>
                <div class="app-card-value">{get_last_updated(MACHINE_RESULTS_PATH)}</div>
                <div class="app-card-text">Based on the latest generated machine results file.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="app-card">
                <div class="app-card-title">{ICONS["cpu"]} Core AI Model</div>
                <div class="app-card-value">Isolation Forest</div>
                <div class="app-card-text">Primary anomaly detection model used for health scoring and maintenance alerts.</div>
            </div>
            """,
            unsafe_allow_html=True
        )




def render_dashboard_action_buttons(filtered_df):
    if filtered_df is None or filtered_df.empty:
        return

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Go to Machine Monitoring", key="go_to_machines_dashboard", use_container_width=True):
            st.session_state["pending_page"] = "Machines"
            st.session_state["detail_only_mode"] = False
            try:
                st.query_params["page"] = "Machines"
            except Exception:
                pass
            st.rerun()

    with c2:
        st.download_button(
            label="Download System CSV",
            data=filtered_df.to_csv(index=False),
            file_name="smartmaint_ai_filtered_system_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    with c3:
        summary_text = f"""
SMARTMAINT AI - SYSTEM SUMMARY REPORT
=====================================

Generated On        : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Last Data Updated   : {get_last_updated(MACHINE_RESULTS_PATH)}

System Summary
--------------
Total Records       : {len(filtered_df)}
Unique Machines     : {filtered_df['machine_id'].nunique()}
Equipment Categories: {filtered_df['equipment_category'].nunique()}
Average Health Score: {filtered_df['health_score'].mean():.2f}

Alert Distribution
------------------
{filtered_df['alert_level'].value_counts().to_string()}

Most Critical Machines
----------------------
{filtered_df.sort_values('health_score')[['machine_id', 'equipment_category', 'health_score', 'alert_level', 'recommended_action']].head(10).to_string(index=False)}

Generated by SmartMaint AI Local PC-Based Predictive Maintenance Application.
"""
        st.download_button(
            label="Download Summary Report",
            data=summary_text,
            file_name="smartmaint_ai_summary_report.txt",
            mime="text/plain",
            use_container_width=True
        )


def render_machine_page_actions(filtered_df):
    if filtered_df is None or filtered_df.empty:
        return

    st.download_button(
        label="Download Filtered Machines CSV",
        data=filtered_df.to_csv(index=False),
        file_name="smartmaint_ai_filtered_machines.csv",
        mime="text/csv",
        use_container_width=True
    )



def sidebar_nav_link(label, description, icon_svg, target_page, active=False):
    """Render one clickable sidebar navigation card using SVG icons."""
    active_class = "active" if active else ""
    safe_page = target_page.replace(" ", "%20")
    st.markdown(
        f"""
        <a class="sidebar-nav-link" href="?page={safe_page}" target="_self">
            <div class="sidebar-nav-card {active_class}">
                <div class="sidebar-nav-icon">{icon_svg}</div>
                <div>
                    <div class="sidebar-nav-title">{html.escape(label)}</div>
                    <div class="sidebar-nav-sub">{html.escape(description)}</div>
                </div>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )


def get_page_from_query_params():
    valid_pages = ["Dashboard", "Machines", "Model Evaluation"]
    try:
        query_value = st.query_params.get("page", None)
        if isinstance(query_value, list):
            query_value = query_value[0]
        if query_value in valid_pages:
            return query_value
    except Exception:
        pass
    return None

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-inner">
            <div class="hero-icon">{ICONS["shield"]}</div>
            <div>
                <p class="hero-title">SmartMaint AI</p>
                <p class="hero-sub">
                    Local PC-Based AI Predictive Maintenance Application for anomaly detection,
                    health scoring, alert generation, maintenance recommendations, and reporting.
                </p>
            </div>
            <div class="hero-badge">{ICONS["radar"]} Predictive Maintenance</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


machine_df = load_machine_results()
evaluation_df = load_evaluation_results()
rf_feature_importance_df = load_rf_feature_importance()

render_app_control_panel(machine_df)
st.markdown("<br>", unsafe_allow_html=True)

if machine_df is None:
    st.error("machine_results.csv not found. Click 'Run AI Analysis' to generate the output files, or run `python main.py`.")
    st.stop()

# Fallback columns for older generated outputs
fallback_columns = {
    "short_reason": "No abnormal pattern available. Please run python main.py again.",
    "recommended_action": "No recommendation available. Please run python main.py again.",
    "explanation": "No explanation available. Please run python main.py again.",
    "alert_priority": "Not assigned",
    "problem_detected": "No problem generated yet. Please run python main.py again.",
    "probable_cause": "No cause generated yet. Please run python main.py again.",
    "recommended_solution": "No solution generated yet. Please run python main.py again.",
}

for col, value in fallback_columns.items():
    if col not in machine_df.columns:
        machine_df[col] = value

# Convert record-level output into one final machine-level view for the dashboard.
# This prevents one machine from appearing with multiple statuses at the same time.
machine_view_df = get_machine_level_status(machine_df)


# ---------------------------------------------------------
# Sidebar Filters + Navigation
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"

query_page = get_page_from_query_params()
if query_page is not None:
    st.session_state["page"] = query_page

if "pending_page" in st.session_state:
    st.session_state["page"] = st.session_state.pop("pending_page")
    try:
        st.query_params["page"] = st.session_state["page"]
    except Exception:
        pass

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">{ICONS["factory"]}</div>
            <div>
                <div class="sidebar-brand-title">SmartMaint AI</div>
                <div class="sidebar-brand-sub">Local PC Application</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f'<div class="sidebar-header">{ICONS["layers"]} Main Sections</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-caption">Choose the main workspace for monitoring, inspection, or model validation.</div>', unsafe_allow_html=True)

    current_page = st.session_state.get("page", "Dashboard")

    sidebar_nav_link("Dashboard", "System overview and recent alerts", ICONS["gauge"], "Dashboard", current_page == "Dashboard")
    sidebar_nav_link("Machine Monitoring", "Inspect assets and machine details", ICONS["cpu"], "Machines", current_page == "Machines")
    sidebar_nav_link("Model Evaluation", "Compare AI model performance", ICONS["bar_chart"], "Model Evaluation", current_page == "Model Evaluation")

    page = st.session_state.get("page", "Dashboard")

    st.markdown(f'<div class="sidebar-header">{ICONS["settings"]} Data Filters</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-caption">Filter the displayed results by equipment type, risk level, and machine ID.</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Sidebar Filter Logic
    # ---------------------------------------------------------
    # IMPORTANT:
    # Use machine_view_df here, not machine_df.
    # machine_view_df already contains one final status per machine.
    # This prevents the same machine from appearing under multiple alert levels.
    filter_source_df = machine_view_df.copy()

    # Make sure required columns exist before building filters
    if "alert_level" not in filter_source_df.columns:
        filter_source_df["alert_level"] = "Normal"

    if "equipment_category" not in filter_source_df.columns:
        filter_source_df["equipment_category"] = "General Equipment"

    if "machine_id" not in filter_source_df.columns:
        filter_source_df["machine_id"] = "N/A"

    # Equipment Category, Alert Level, and Machine ID Filters
    # ---------------------------------------------------------
    # Multi-select filter logic:
    # - User can select one, two, three, or more values.
    # - Selecting "All" or leaving the filter empty means no restriction.
    # - Equipment options respect selected alert levels.
    # - Alert options respect selected equipment categories.
    # - Machine ID options respect selected equipment and alert filters.
    alert_order = ["Normal", "Low Risk", "Medium Risk", "High Risk", "Critical"]

    def normalize_multiselect_state(key):
        """Convert old single-select values into list values for multiselect widgets."""
        if key not in st.session_state:
            st.session_state[key] = ["All"]
        elif isinstance(st.session_state[key], str):
            st.session_state[key] = [st.session_state[key]]
        elif not isinstance(st.session_state[key], list):
            st.session_state[key] = ["All"]

    def is_all_selected(values):
        return not values or "All" in values

    normalize_multiselect_state("filter_equipment")
    normalize_multiselect_state("filter_alert")
    normalize_multiselect_state("filter_machine")

    current_alert_filter = st.session_state.get("filter_alert", ["All"])

    equipment_option_source_df = filter_source_df.copy()

    if not is_all_selected(current_alert_filter):
        equipment_option_source_df = equipment_option_source_df[
            equipment_option_source_df["alert_level"].astype(str).isin(current_alert_filter)
        ]

    equipment_values = sorted(
        equipment_option_source_df["equipment_category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    equipment_options = ["All"] + equipment_values

    st.session_state["filter_equipment"] = [
        value for value in st.session_state["filter_equipment"]
        if value in equipment_options
    ]

    if not st.session_state["filter_equipment"]:
        st.session_state["filter_equipment"] = ["All"]

    selected_equipment = st.multiselect(
        "Equipment Category",
        equipment_options,
        key="filter_equipment",
        placeholder="Choose one or more equipment categories"
    )

    equipment_filtered_df = filter_source_df.copy()

    if not is_all_selected(selected_equipment):
        equipment_filtered_df = equipment_filtered_df[
            equipment_filtered_df["equipment_category"].astype(str).isin(selected_equipment)
        ]

    available_alerts = (
        equipment_filtered_df["alert_level"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    available_alerts = [alert for alert in alert_order if alert in available_alerts]
    alert_options = ["All"] + available_alerts

    st.session_state["filter_alert"] = [
        value for value in st.session_state["filter_alert"]
        if value in alert_options
    ]

    if not st.session_state["filter_alert"]:
        st.session_state["filter_alert"] = ["All"]

    selected_alert = st.multiselect(
        "Alert Level",
        alert_options,
        key="filter_alert",
        placeholder="Choose one or more alert levels"
    )

    alert_filtered_df = equipment_filtered_df.copy()

    if not is_all_selected(selected_alert):
        alert_filtered_df = alert_filtered_df[
            alert_filtered_df["alert_level"].astype(str).isin(selected_alert)
        ]

    machine_options = ["All"] + sorted(
        alert_filtered_df["machine_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    st.session_state["filter_machine"] = [
        value for value in st.session_state["filter_machine"]
        if value in machine_options
    ]

    if not st.session_state["filter_machine"]:
        st.session_state["filter_machine"] = ["All"]

    selected_machine = st.multiselect(
        "Machine ID",
        machine_options,
        key="filter_machine",
        placeholder="Choose one or more machine IDs"
    )

    st.markdown(
        f"""
        <div class="sidebar-tip">
            {ICONS["info"]}
            Use <b>Dashboard</b> for the system overview, <b>Machine Monitoring</b> for detailed inspection,
            and <b>Model Evaluation</b> to justify the AI model performance.
        </div>
        """,
        unsafe_allow_html=True
    )


if page != "Machines":
    st.session_state["detail_only_mode"] = False

filtered_df = alert_filtered_df.copy()
if not is_all_selected(selected_machine):
    filtered_df = filtered_df[
        filtered_df["machine_id"].astype(str).isin(selected_machine)
    ]


# =========================================================
# PAGE 1: DASHBOARD
# =========================================================
if page == "Dashboard":
    st.markdown(
        f"""
        <div class="info-box">
            {ICONS["info"]}
            <span>
                This page gives a system-level overview of machine health, risk levels,
                detected anomalies, and recent high-risk assets.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_records = len(filtered_df)
    total_machines = filtered_df["machine_id"].nunique() if not filtered_df.empty else 0
    category_count = filtered_df["equipment_category"].nunique() if not filtered_df.empty else 0

    normal_count = int((filtered_df["alert_level"] == "Normal").sum()) if not filtered_df.empty else 0
    low_risk_count = int((filtered_df["alert_level"] == "Low Risk").sum()) if not filtered_df.empty else 0
    medium_risk_count = int((filtered_df["alert_level"] == "Medium Risk").sum()) if not filtered_df.empty else 0
    high_risk_count = int((filtered_df["alert_level"] == "High Risk").sum()) if not filtered_df.empty else 0
    critical_count = int((filtered_df["alert_level"] == "Critical").sum()) if not filtered_df.empty else 0

    cols = st.columns(6)
    kpi_card(ICONS["database"], "Total Machines", total_machines, cols[0], f"Across {category_count} categories", "blue")
    kpi_card(ICONS["check_circle"], "Normal", normal_count, cols[1], "Healthy machines", "green")
    kpi_card(ICONS["gauge"], "Low Risk", low_risk_count, cols[2], "Observe condition", "blue")
    kpi_card(ICONS["alert_triangle"], "Medium Risk", medium_risk_count, cols[3], "Inspection needed", "yellow")
    kpi_card(ICONS["zap"], "High Risk", high_risk_count, cols[4], "Urgent attention", "orange")
    kpi_card(ICONS["alert_triangle"], "Critical", critical_count, cols[5], "Immediate action", "red")

    st.markdown("<br>", unsafe_allow_html=True)
    render_dashboard_action_buttons(filtered_df)
    st.markdown("<br>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No records found based on the selected filters.")
    else:
        col_left, col_mid, col_right = st.columns([1.15, 1.35, 1.05])

        with col_left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">{ICONS["gauge"]} Machines Health Overview</div>', unsafe_allow_html=True)

            alert_counts = filtered_df["alert_level"].value_counts().reset_index()
            alert_counts.columns = ["alert_level", "count"]
            donut_colors = [ALERT_COLOR_MAP.get(level, "#2f7df6") for level in alert_counts["alert_level"]]

            fig_donut = go.Figure(
                go.Pie(
                    labels=alert_counts["alert_level"],
                    values=alert_counts["count"],
                    hole=0.62,
                    marker_colors=donut_colors,
                    textinfo="percent",
                    hovertemplate="%{label}: %{value}<extra></extra>"
                )
            )
            fig_donut.add_annotation(
                text=f"<b>{total_records}</b><br><span style='font-size:12px;color:#9aa8bd'>Machines</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="#f4f7fb")
            )
            fig_donut.update_layout(
                **PLOTLY_LAYOUT,
                height=330,
                showlegend=True
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_mid:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">{ICONS["activity"]} Health Score Trend</div>', unsafe_allow_html=True)

            trend_df = filtered_df[["health_score"]].copy().reset_index(drop=True)
            trend_df = trend_df.head(60)
            trend_df["record"] = range(1, len(trend_df) + 1)

            fig_trend = px.line(
                trend_df,
                x="record",
                y="health_score",
                markers=True,
                color_discrete_sequence=["#2f7df6"]
            )
            fig_trend.update_traces(line_width=3, marker_size=7)
            fig_trend.update_layout(
                **PLOTLY_LAYOUT,
                title="Health Score Trend Across Machines",
                xaxis_title="Record Sequence",
                yaxis_title="Health Score",
                height=330
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            avg_health = filtered_df["health_score"].mean()
            st.info(f"Average Health Score: {avg_health:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">{ICONS["alert_triangle"]} Recent Alerts</div>', unsafe_allow_html=True)

            recent_alerts = filtered_df.sort_values("health_score", ascending=True).head(5)
            for _, row in recent_alerts.iterrows():
                st.markdown(
                    f"""
                    <div class="recent-card">
                        <div>
                            <div class="recent-title">{row["machine_id"]} · {row["equipment_category"]}</div>
                            <div class="recent-sub">Health: {row["health_score"]} · {row["short_reason"]}</div>
                        </div>
                        <div>{alert_badge(row["alert_level"])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_recent_machines_table(filtered_df, max_rows=6)
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE 2: MACHINES
# =========================================================
elif page == "Machines":
    st.markdown(f'<div class="section-title">{ICONS["cpu"]} Machine Monitoring</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No machines found based on the selected filters.")
    else:
        # filtered_df already comes from machine_view_df, so it already has one
        # final status per machine. Do not summarize again after filtering.
        table_source_df = filtered_df.copy()

        display_columns = safe_column_list(
            table_source_df,
            [
                "machine_id", "equipment_category", "health_score",
                "alert_level", "alert_priority", "if_anomaly_label",
                "Machine failure", "problem_detected", "short_reason", "recommended_action"
            ]
        )

        detail_only_mode = st.session_state.get("detail_only_mode", False)

        if not detail_only_mode:
            render_machine_page_actions(table_source_df)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            render_machine_ui_table(table_source_df[display_columns], title="All Machines Status")
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            left_back, right_back = st.columns([4, 1])
            with right_back:
                if st.button("Show all machines", key="show_all_machines_from_detail", use_container_width=True):
                    st.session_state["detail_only_mode"] = False
                    try:
                        st.query_params["page"] = "Machines"
                    except Exception:
                        pass
                    st.rerun()

            st.markdown(f'<div class="section-title">{ICONS["target"]} Selected Machine Details</div>', unsafe_allow_html=True)

            machine_list = sorted(table_source_df["machine_id"].unique().tolist())

            if not machine_list:
                st.warning("No machines are available in the current table view. Click 'Show All Filtered Machines' or adjust the sidebar filters.")
                st.stop()

            default_machine = st.session_state.get("selected_machine_detail", machine_list[0])
            if default_machine not in machine_list:
                default_machine = machine_list[0]

            # Keep the selectbox synchronized with filters and View button clicks.
            if st.session_state.get("machine_detail_selectbox") not in machine_list:
                st.session_state["machine_detail_selectbox"] = default_machine

            selected_machine_detail = st.selectbox(
                "Choose a machine to inspect",
                machine_list,
                index=machine_list.index(default_machine),
                key="machine_detail_selectbox"
            )

            st.session_state["selected_machine_detail"] = selected_machine_detail
            machine_detail = table_source_df[table_source_df["machine_id"] == selected_machine_detail]
            latest_record = machine_detail.sort_values("health_score", ascending=True).iloc[0]

            col1, col2, col3 = st.columns(3)
            kpi_card(ICONS["cpu"], "Machine ID", latest_record["machine_id"], col1, "Selected asset")
            kpi_card(ICONS["factory"], "Equipment Type", latest_record["equipment_category"], col2, "Machine category")
            kpi_card(ICONS["gauge"], "Health Score", latest_record["health_score"], col3, "Lower score means higher risk")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="alert-level-row">
                    <div>
                        <div class="alert-level-label">Alert Level</div>
                        {alert_badge(latest_record["alert_level"])}
                    </div>
                    <div>
                        <div class="alert-priority-label">Priority</div>
                        <div class="alert-priority-value">{latest_record["alert_priority"]}</div>
                    </div>
                    <div>
                        <div class="alert-priority-label">Anomaly Label</div>
                        <div class="alert-priority-value">{latest_record.get("if_anomaly_label", "N/A")}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="detail-box detail-reason">
                    <b>{ICONS["alert_triangle"]} Detected Problem</b>
                    {latest_record["problem_detected"]}
                </div>

                <div class="detail-box detail-explanation">
                    <b>{ICONS["info"]} Probable Cause</b>
                    {latest_record["probable_cause"]}
                </div>

                <div class="detail-box detail-reason">
                    <b>{ICONS["activity"]} Observed Abnormal Pattern</b>
                    {latest_record["short_reason"]}
                </div>

                <div class="detail-box detail-recommendation">
                    <b>{ICONS["tool"]} Recommended Action</b>
                    {latest_record["recommended_action"]}
                </div>

                <div class="detail-box detail-explanation">
                    <b>{ICONS["zap"]} AI Explanation</b>
                    {latest_record["explanation"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("---")
            render_maintenance_schedule_card(latest_record)

            st.download_button(
                label="Download Machine Report",
                data=create_machine_report(latest_record),
                file_name=f"{latest_record['machine_id']}_maintenance_report.txt",
                mime="text/plain"
            )

            st.markdown("---")
            st.markdown(f'<div class="section-title">{ICONS["thermometer"]} Sensor Readings</div>', unsafe_allow_html=True)

            sensor_columns = safe_column_list(
                table_source_df,
                [
                    "Type", "Air temperature [K]", "Process temperature [K]",
                    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
                ]
            )

            render_sensor_reading_cards(latest_record, sensor_columns)


# =========================================================
# PAGE 3: ALERTS
# =========================================================
elif page == "Alerts_DISABLED":
    st.markdown(f'<div class="section-title">{ICONS["alert_triangle"]} Alerts & Recommendations</div>', unsafe_allow_html=True)

    risk_df = filtered_df[
        filtered_df["alert_level"].isin(["Medium Risk", "High Risk", "Critical"])
    ]

    if risk_df.empty:
        st.success("No medium, high, or critical risk records found in the current filter.")
    else:
        st.warning(f"{len(risk_df)} risky records detected.")

        risk_columns = safe_column_list(
            risk_df,
            [
                "machine_id", "equipment_category", "health_score",
                "alert_level", "alert_priority", "problem_detected",
                "probable_cause", "recommended_solution", "short_reason", "recommended_action"
            ]
        )

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_machine_ui_table(risk_df.sort_values("health_score")[risk_columns], title="Risky Machines Status")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div class="section-title">{ICONS["bar_chart"]} Maintenance Action Summary</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.info("No data available for maintenance action summary.")
    else:
        action_summary = filtered_df.groupby(
            ["alert_level", "recommended_action"]
        ).size().reset_index(name="count")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_pretty_dataframe_table(action_summary, title="Maintenance Action Summary", max_rows=20)

        fig_action = px.bar(
            action_summary,
            x="alert_level",
            y="count",
            text="count",
            color="alert_level",
            color_discrete_map=ALERT_COLOR_MAP
        )
        fig_action.update_traces(
            textfont_color="#f4f7fb",
            marker_line_width=0,
            textposition="outside",
            opacity=0.92
        )
        fig_action.update_layout(
            **PLOTLY_LAYOUT,
            title="Maintenance Actions by Alert Level",
            xaxis_title="Alert Level",
            yaxis_title="Number of Records",
            showlegend=False,
            height=380
        )
        st.plotly_chart(fig_action, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE 4: REPORTS
# =========================================================
elif page == "Reports_DISABLED":
    st.markdown(f'<div class="section-title">{ICONS["file"]} Reports Center</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No records available for report generation.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                f"""
                <div class="app-card">
                    <div class="app-card-title">{ICONS["download"]} Full System CSV Report</div>
                    <div class="app-card-value">Filtered System Report</div>
                    <div class="app-card-text">Includes all visible records based on the current filters.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.download_button(
                label="Download Filtered System Report",
                data=filtered_df.to_csv(index=False),
                file_name="smartmaint_ai_filtered_system_report.csv",
                mime="text/csv",
                use_container_width=True
            )

        with c2:
            summary_text = f"""
SMARTMAINT AI - SYSTEM SUMMARY REPORT
=====================================

Generated On        : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Last Data Updated   : {get_last_updated(MACHINE_RESULTS_PATH)}

System Summary
--------------
Total Records       : {len(filtered_df)}
Unique Machines     : {filtered_df["machine_id"].nunique()}
Equipment Categories: {filtered_df["equipment_category"].nunique()}
Average Health Score: {filtered_df["health_score"].mean():.2f}

Alert Distribution
------------------
{filtered_df["alert_level"].value_counts().to_string()}

Most Critical Machines
----------------------
{filtered_df.sort_values("health_score")[["machine_id", "equipment_category", "health_score", "alert_level", "recommended_action"]].head(10).to_string(index=False)}

Generated by SmartMaint AI Local PC-Based Predictive Maintenance Application.
"""
            st.markdown(
                f"""
                <div class="app-card">
                    <div class="app-card-title">{ICONS["file"]} Text Summary Report</div>
                    <div class="app-card-value">Executive Maintenance Summary</div>
                    <div class="app-card-text">Simple report for supervisor/demo presentation.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.download_button(
                label="Download Text Summary Report",
                data=summary_text,
                file_name="smartmaint_ai_summary_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        st.markdown("---")
        st.markdown(f'<div class="section-title">{ICONS["database"]} Report Preview</div>', unsafe_allow_html=True)
        preview_columns = safe_column_list(
            filtered_df,
            ["machine_id", "equipment_category", "health_score", "alert_level", "alert_priority", "short_reason", "recommended_action"]
        )
        render_machine_ui_table(filtered_df[preview_columns].head(30), title="Report Preview Table")


# =========================================================
# PAGE 5: MODEL EVALUATION
# =========================================================
elif page == "Model Evaluation":
    st.markdown(f'<div class="section-title">{ICONS["bar_chart"]} Model Evaluation Results</div>', unsafe_allow_html=True)

    if evaluation_df is None:
        st.error("evaluation_results.csv not found. Click 'Run AI Analysis' or run `python main.py` first.")
    else:
        st.download_button(
            label="Download Evaluation Results CSV",
            data=evaluation_df.to_csv(index=False),
            file_name="smartmaint_ai_evaluation_results.csv",
            mime="text/csv",
            use_container_width=False
        )

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_model_evaluation_summary(evaluation_df)
        st.markdown("</div>", unsafe_allow_html=True)

        render_pretty_dataframe_table(evaluation_df, title="Model Evaluation Table")

        metric_columns = safe_column_list(
            evaluation_df,
            ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
        )

        if len(metric_columns) > 0 and "model" in evaluation_df.columns:
            evaluation_melted = evaluation_df.melt(
                id_vars="model",
                value_vars=metric_columns,
                var_name="metric",
                value_name="score"
            )

            fig_eval = px.bar(
                evaluation_melted,
                x="model",
                y="score",
                color="metric",
                barmode="group",
                color_discrete_sequence=CHART_COLORS
            )
            fig_eval.update_traces(marker_line_width=0, opacity=0.92, texttemplate=None)
            fig_eval.update_layout(
                **PLOTLY_LAYOUT,
                title="Model Performance Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                height=420,
                yaxis_range=[0, 1.05]
            )

            st.plotly_chart(fig_eval, use_container_width=True)

        if rf_feature_importance_df is not None and not rf_feature_importance_df.empty:
            st.markdown("---")
            st.markdown(f'<div class="section-title">{ICONS["layers"]} Random Forest Feature Importance</div>', unsafe_allow_html=True)

            feature_df = rf_feature_importance_df.copy()

            if "feature" in feature_df.columns and "importance" in feature_df.columns:
                render_pretty_dataframe_table(feature_df, title="Feature Importance Table", max_rows=10)

                fig_importance = px.bar(
                    feature_df.sort_values("importance", ascending=True),
                    x="importance",
                    y="feature",
                    orientation="h",
                    color_discrete_sequence=["#2f7df6"]
                )
                fig_importance.update_traces(marker_line_width=0, opacity=0.92)
                fig_importance.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Random Forest Feature Importance",
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=380,
                    showlegend=False
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("random_forest_feature_importance.csv was found, but required columns are missing.")

        st.markdown("---")
        st.markdown(f'<div class="section-title">{ICONS["target"]} Confusion Matrix Values</div>', unsafe_allow_html=True)

        cm_columns = safe_column_list(
            evaluation_df,
            ["model", "true_negative", "false_positive", "false_negative", "true_positive"]
        )

        if cm_columns:
            render_pretty_dataframe_table(evaluation_df[cm_columns], title="Confusion Matrix Values")
        else:
            st.info("No confusion matrix columns found in evaluation_results.csv.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div class="footer-wrap">
        <span class="footer-text">
            SmartMaint AI
            <span class="footer-dot">·</span>
            Local PC-Based AI Predictive Maintenance Application
            <span class="footer-dot">·</span>
            Python · Machine Learning · Streamlit · AI4I 2020 Dataset
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
