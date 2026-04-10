import os
import re
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="AURORA 2.0 - ISRO Monitoring", 
    layout="wide", 
    initial_sidebar_state="collapsed",)

if 'current_mine' not in st.session_state:
    st.session_state.current_mine = None

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Animated Background Grid */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Main Content Container */
    .main > div {
        position: relative;
        z-index: 1;
    }
    
    /* Main Title - Futuristic */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 25%, #00ffff 50%, #00d4ff 75%, #0099ff 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: 8px;
        text-shadow: 0 0 40px rgba(0, 212, 255, 0.5);
        animation: gradient-shift 3s ease infinite;
        position: relative;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .main-title::after {
        content: '🛰️';
        position: absolute;
        right: -60px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 3rem;
        animation: satellite-float 3s ease-in-out infinite;
    }
    
    @keyframes satellite-float {
        0%, 100% { transform: translateY(-50%) translateX(0); }
        50% { transform: translateY(-50%) translateX(10px); }
    }
    
    /* Subtitle */
    .subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.4rem;
        color: #7dd3fc;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 500;
        letter-spacing: 3px;
        text-transform: uppercase;
        text-shadow: 0 0 20px rgba(125, 211, 252, 0.3);
    }
    
    /* Section Headers - Neon Style */
    .section-header {
        background: linear-gradient(90deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 153, 255, 0.1) 100%);
        border: 2px solid #00d4ff;
        border-left: 6px solid #00d4ff;
        color: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 15px;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 3rem 0 2rem 0;
        box-shadow: 
            0 0 30px rgba(0, 212, 255, 0.2),
            inset 0 0 20px rgba(0, 212, 255, 0.05);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: 'Orbitron', sans-serif;
        position: relative;
        overflow: hidden;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.2), transparent);
        animation: scan 3s linear infinite;
    }
    
    @keyframes scan {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Image Containers */
    .image-container {
        background: rgba(15, 20, 35, 0.8);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            0 0 20px rgba(0, 212, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .image-container:hover {
        border-color: #00d4ff;
        box-shadow: 
            0 12px 48px rgba(0, 0, 0, 0.4),
            0 0 40px rgba(0, 212, 255, 0.3);
        transform: translateY(-5px);
    }
    
    /* Image Title */
    .image-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #00d4ff;
        margin-bottom: 1rem;
        padding: 0.8rem 1.2rem;
        background: linear-gradient(90deg, rgba(0, 212, 255, 0.15) 0%, transparent 100%);
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    /* Metric Cards - Holographic Style */
    [data-testid="stMetricValue"] {
        font-size: 2.8rem;
        font-weight: 900;
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
        color: #7dd3fc !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.05) 0%, rgba(0, 153, 255, 0.05) 100%);
        border: 2px solid rgba(0, 212, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.2),
            inset 0 0 20px rgba(0, 212, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        border-color: #00d4ff;
        box-shadow: 
            0 12px 48px rgba(0, 0, 0, 0.3),
            inset 0 0 30px rgba(0, 212, 255, 0.1);
        transform: translateY(-3px);
    }
    
    /* Sidebar - Cyber Theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e1f 0%, #1a1f35 50%, #0f1428 100%);
        border-right: 2px solid rgba(0, 212, 255, 0.3);
        box-shadow: 5px 0 30px rgba(0, 212, 255, 0.1);
    }
    
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 212, 255, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 212, 255, 0.02) 1px, transparent 1px);
        background-size: 30px 30px;
        pointer-events: none;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e0f2fe;
    }
    
    [data-testid="stSidebar"] h2 {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.6rem;
        color: #00d4ff !important;
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
        letter-spacing: 2px;
    }
    
    [data-testid="stSidebar"] h3 {
        font-family: 'Orbitron', sans-serif;
        color: #7dd3fc !important;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        letter-spacing: 1px;
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: #7dd3fc !important;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stSidebar"] .stMetric label {
        color: #94a3b8 !important;
        font-size: 0.9rem;
    }
    
    [data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 2rem;
    }
    
    /* Info Box in Sidebar */
    .sidebar-info-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 153, 255, 0.05) 100%);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #e0f2fe;
        font-size: 0.95rem;
        line-height: 1.8;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.2),
            inset 0 0 15px rgba(0, 212, 255, 0.05);
    }
    
    /* Alert Box in Sidebar */
    .sidebar-alert-box {
        background: linear-gradient(135deg, rgba(255, 107, 0, 0.15) 0%, rgba(255, 153, 0, 0.1) 100%);
        border: 2px solid rgba(255, 153, 0, 0.4);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        color: #ffd700;
        font-size: 0.95rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 0 20px rgba(255, 153, 0, 0.2);
        animation: pulse-alert 2s ease-in-out infinite;
    }
    
    @keyframes pulse-alert {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 153, 0, 0.2); }
        50% { box-shadow: 0 0 30px rgba(255, 153, 0, 0.4); }
    }
    
    /* Tabs - Modern Cyber Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(10, 14, 31, 0.5);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 14px 28px;
        font-weight: 700;
        font-size: 1.1rem;
        font-family: 'Rajdhani', sans-serif;
        letter-spacing: 1px;
        background: rgba(0, 212, 255, 0.05);
        border: 2px solid rgba(0, 212, 255, 0.2);
        border-radius: 10px;
        color: #7dd3fc;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 212, 255, 0.1);
        border-color: #00d4ff;
        color: #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 153, 255, 0.2) 100%);
        border-color: #00d4ff !important;
        color: #00d4ff !important;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.4);
    }
    
    /* Buttons - Futuristic */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 153, 255, 0.2) 100%);
        border: 2px solid #00d4ff;
        border-radius: 10px;
        color: #00d4ff;
        font-weight: 700;
        font-size: 1rem;
        font-family: 'Rajdhani', sans-serif;
        letter-spacing: 1px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.3) 0%, rgba(0, 153, 255, 0.3) 100%);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        transform: translateY(-2px);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.15) 0%, rgba(0, 153, 255, 0.15) 100%);
        border: 2px solid rgba(0, 212, 255, 0.5);
        border-radius: 10px;
        color: #00d4ff;
        font-weight: 600;
        font-family: 'Rajdhani', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 25px rgba(0, 212, 255, 0.4);
        transform: translateY(-2px);
    }
    
    /* Dataframes - High-tech */
    .dataframe {
        font-size: 0.95rem;
        font-family: 'Rajdhani', sans-serif;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 153, 255, 0.2) 100%);
        color: #00d4ff !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 2px solid #00d4ff;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(0, 212, 255, 0.05);
    }
    
    /* Expander - Sleek Design */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(0, 212, 255, 0.1) 0%, transparent 100%);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        color: #00d4ff !important;
        font-weight: 700;
        font-family: 'Rajdhani', sans-serif;
        letter-spacing: 1px;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        background: rgba(10, 14, 31, 0.8);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        color: #e0f2fe;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
    }
    
    /* Multi-select */
    .stMultiSelect > div > div {
        background: rgba(10, 14, 31, 0.8);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
    }
    
    /* Radio Buttons */
    .stRadio > div {
        background: rgba(10, 14, 31, 0.5);
        border: 2px solid rgba(0, 212, 255, 0.2);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.5), transparent);
        margin: 2rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 0 2rem 0;
        color: #64748b;
        font-size: 0.95rem;
        border-top: 2px solid rgba(0, 212, 255, 0.2);
        margin-top: 4rem;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Success/Info/Warning/Error Messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 10px;
        border-left-width: 6px;
    }
    
    /* Images */
    img {
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    img:hover {
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4), 0 0 30px rgba(0, 212, 255, 0.2);
        transform: scale(1.01);
    }
    
    /* Caption */
    .stCaption {
        color: #94a3b8 !important;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Code blocks */
    code {
        background: rgba(10, 14, 31, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 8px;
        color: #00d4ff !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(10, 14, 31, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d4ff 0%, #0099ff 100%);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #00ffff 0%, #00d4ff 100%);
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #00d4ff !important;
    }
</style>
""", unsafe_allow_html=True)

MINE_DATA_DIR = "Mine Data"

# HELPER FUNCTIONS

def detect_all_mines():
    """Detect available Mine IDs from folder structure"""
    mines = []
    pattern = re.compile(r"Mine_(\d+)_Data", re.IGNORECASE)

    if not os.path.exists(MINE_DATA_DIR):
        return []

    for name in os.listdir(MINE_DATA_DIR):
        m = pattern.match(name)
        if m:
            mines.append(int(m.group(1)))

    return sorted(mines)

def mine_folder(mid: int):
    return os.path.join(MINE_DATA_DIR, f"Mine_{mid}_Data")

def outputs_dir(mid: int):
    return os.path.join(mine_folder(mid), "Outputs")

def alerts_file(mid: int):
    return os.path.join(outputs_dir(mid), f"mine_{mid}_alerts.log")

def list_files(folder: str):
    if not os.path.exists(folder):
        return []
    return sorted(os.listdir(folder))



def parse_alerts(log_path):
    """Parse alerts log file with multi-line format support"""
    if not os.path.exists(log_path):
        return None
    
    alerts = []
    current_alert = None
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check if this is a header line (starts with [date])
                if line.startswith('['):
                    # Save previous alert if exists
                    if current_alert:
                        alerts.append(current_alert)
                    
                    # Parse header: [date] | Mine X | LEVEL Y | TYPE
                    parts = [p.strip() for p in line.split('|')]
                    
                    if len(parts) >= 4:
                        date = parts[0].strip('[]')
                        mine_id = parts[1]
                        level = parts[2]
                        alert_type = parts[3]
                        
                        # Map LEVEL to proper severity
                        if 'LEVEL 1' in level:
                            severity = 'WARNING'
                        elif 'LEVEL 2' in level:
                            severity = 'CRITICAL'
                        else:
                            severity = 'INFO'
                        
                        current_alert = {
                            'Timestamp': date,
                            'Mine': mine_id,
                            'Level': level,
                            'Type': alert_type,
                            'Severity': severity,
                            'Details': []
                        }
                    else:
                        # Malformed header, skip
                        current_alert = None
                else:
                    # This is a detail line for the current alert
                    if current_alert:
                        current_alert['Details'].append(line)
            
            # Don't forget the last alert
            if current_alert:
                alerts.append(current_alert)
        
        if alerts:
            # Convert to clean format
            clean_alerts = []
            for alert in alerts:
                details_text = '\n'.join(alert['Details'])
                clean_alerts.append({
                    'Timestamp': alert['Timestamp'],
                    'Severity': alert['Severity'],
                    'Type': alert['Type'],
                    'Message': details_text
                })
            
            return pd.DataFrame(clean_alerts)
        return None
    
    except Exception as e:
        st.error(f"Error reading alerts: {e}")
        return None

def categorize_images(images):
    """Intelligently categorize images into logical groups"""
    categories = {
        "spatial": [],
        "progress": [],
        "area_analysis": [],
        "growth_metrics": [],
        "no_go_violations": [],
        "other": []
    }
    
    for img in images:
        img_lower = img.lower()
        
        if "spatialmap" in img_lower or "percent" in img_lower:
            categories["spatial"].append(img)
        elif "progress" in img_lower: 
            categories["progress"].append(img)
        elif "no_go" in img_lower:
            categories["no_go_violations"].append(img)
        elif any(x in img_lower for x in ["areavstime", "candidatearea", "comparision"]):
            categories["area_analysis"].append(img)
        elif any(x in img_lower for x in ["normalized", "growthrate", "firstseen"]):
            categories["growth_metrics"].append(img)
        else:
            categories["other"].append(img)
    
    return categories

def sort_spatial_maps(images):
    """Sort spatial maps chronologically - 0% to 100%"""
    def get_order(img):
        img_lower = img.lower()
        if "_0percent" in img_lower:
            return 0
        elif "_25percent" in img_lower:
            return 1
        elif "_50percent" in img_lower:
            return 2
        elif "_75percent" in img_lower:
            return 3
        elif "_100percent" in img_lower:
            return 4
        return 999
    
    return sorted(images, key=get_order)

def display_image_with_style(img_path, title, caption=""):
    """Display image with consistent professional styling"""
    st.markdown(f"""
    <div class="image-container">
        <div class="image-title">📡 {title}</div>
    </div>
    """, unsafe_allow_html=True)
    if caption:
        st.caption(caption)
    st.image(img_path, use_container_width=True)

# MAIN UI

# Header with animation
st.markdown('<div class="main-title">AURORA 2.0</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Underground Resource Observation & Remote Analysis System</div>', unsafe_allow_html=True)

available_mines = detect_all_mines()

if not available_mines:
    st.error(" No mine data found. Please ensure Mine Data folder exists with proper structure.")
    st.info("Expected structure: `Mine Data/Mine_X_Data/Outputs/`")
    st.stop()

# SIDEBAR

with st.sidebar:
    st.markdown("## AURORA 2.0")
    st.markdown("**Satellite Monitoring System**")
    st.markdown("---")
    
    st.markdown("### Mine Selection")
    mine_id = st.selectbox(
        "Select Mine Site",
        available_mines,
        format_func=lambda x: f"Mine Site {x}",
        help="Choose a mine site to view monitoring data"
    )
    
    st.session_state.current_mine = mine_id

    st.markdown("---")
    
    # Mine info card
    st.markdown("### Site Information")

    st.markdown(f"""
    <div class="sidebar-info-box">
    <b>Data Location:</b> <code>Mine_{mine_id}_Data</code>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Quick stats
    out_dir = outputs_dir(mine_id)
    intensity_csv = os.path.join(out_dir, f"mine_{mine_id}_ExcavationIntensity.csv")
    if os.path.exists(out_dir):
        files = list_files(out_dir)
        images = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        csvs = [f for f in files if f.lower().endswith(".csv")]
        
        st.markdown("### Available Data")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Maps", len(images))
        with col2:
            st.metric("Reports", len(csvs))

    if os.path.exists(intensity_csv):
        df = pd.read_csv(intensity_csv)
        latest_area = df["excavated_area_m2"].iloc[-1]

        st.metric(
            " Current Excavated Area",
            f"{latest_area:,.0f} m²"
        )
        
        # Check for alerts
        alert_log = alerts_file(mine_id)
        if os.path.exists(alert_log):
            alerts_df = parse_alerts(alert_log)
            if alerts_df is not None and not alerts_df.empty:
                st.markdown("---")
                st.markdown(f"""
                <div class="sidebar-alert-box">
                 {len(alerts_df)} Alert(s) Detected
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f" Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption("🇮🇳 ISRO Authorized System")

# LOAD DATA

out_dir = outputs_dir(mine_id)
alert_log = alerts_file(mine_id)

if not os.path.exists(out_dir):
    st.error(f" Outputs folder not found for Mine Site {mine_id}")
    st.info(f" Expected location: `{out_dir}`")
    st.stop()

files = list_files(out_dir)
if not files:
    st.warning(f" No output files found for Mine Site {mine_id}")
    st.info(" Please run the monitoring system to generate analysis results.")
    st.stop()

images = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
csvs = [f for f in files if f.lower().endswith(".csv")]

# MAIN TABS

tab1, tab2, tab3, tab4 = st.tabs([
    " Spatial Analysis", 
    " Temporal Metrics", 
    " Data Reports", 
    " Alert System"
])

# TAB 1: SPATIAL MAPS

with tab1:
    st.markdown('<div class="section-header">Spatial Excavation Analysis</div>', unsafe_allow_html=True)
    
    categories = categorize_images(images)
    spatial_maps = sort_spatial_maps(categories["spatial"])
    
    if not spatial_maps:
        st.info("No spatial analysis maps available for this mine site.")
    else:
        st.markdown("""
        ** Monitoring Timeline:** The following maps display the spatial distribution and temporal evolution 
        of excavated areas throughout the monitoring period, from initial detection to current status.
        """)
        
        st.markdown("---")
        
        for img in spatial_maps:
            path = os.path.join(out_dir, img)

            if "_0percent" in img.lower():
                stage = "0% – Initial State "
            elif "_25percent" in img.lower():
                stage = "25% – Early Development "
            elif "_50percent" in img.lower():
                stage = "50% – Mid-point "
            elif "_75percent" in img.lower():
                stage = "75% – Advanced Stage "
            elif "_100percent" in img.lower():
                stage = "100% – Current Status "
            else:
                stage = "Monitoring Stage "

            display_image_with_style(
                path,
                title=f"Stage: {stage}",
                caption=f" File: {img}"
            )

            st.markdown("---")

        progress_imgs = categories["progress"]
        if progress_imgs:
            st.markdown('<div class="section-header"> Complete Temporal Evolution</div>', unsafe_allow_html=True)
            st.markdown("""
            **Full Timeline Grid:** Comprehensive view of excavation progression across the entire monitoring period.
            """)
            
            for img in progress_imgs:
                path = os.path.join(out_dir, img)
                display_image_with_style(
                    path,
                    title="Excavation Progress Overview (0-100 Percentile)",
                    caption=f" File: {img}"
                )
        
        # Download section
        st.markdown("---")
        with st.expander("Download Spatial Maps"):
            st.markdown("** Available Maps for Download:**")
            cols = st.columns(3)
            
            # Download spatial maps
            for idx, img in enumerate(spatial_maps):
                with cols[idx % 3]:
                    with open(os.path.join(out_dir, img), "rb") as f:
                        st.download_button(
                            label=f"⬇️ {img.split('_')[-1].replace('.png', '')}",
                            data=f,
                            file_name=img,
                            mime="image/png",
                            use_container_width=True
                        )
            
            # Download progress grid images
            if progress_imgs:
                st.markdown("---")
                st.markdown("**Progress Grid:**")
                for img in progress_imgs:
                    with open(os.path.join(out_dir, img), "rb") as f:
                        st.download_button(
                            label=f"⬇️ {img}",
                            data=f,
                            file_name=img,
                            mime="image/png",
                            use_container_width=True
                        )

# TAB 2: TIME-SERIES & METRICS

with tab2:
    categories = categorize_images(images)
    
    # Area Analysis Section
    if categories["area_analysis"]:
        st.markdown('<div class="section-header">Temporal Area Analysis</div>', unsafe_allow_html=True)
        st.markdown("**Time-series analysis of excavation area growth and development patterns**")
        st.markdown("---")
        
        for img in categories["area_analysis"]:
            path = os.path.join(out_dir, img)
            title = os.path.splitext(img)[0].replace("_", " ").title()
            caption = f"Analysis: {img}"
            
            display_image_with_style(path, title, caption)
            st.markdown("---")
    
    # Growth Metrics Section
    if categories["growth_metrics"]:
        st.markdown('<div class="section-header">Growth & Development Metrics</div>', unsafe_allow_html=True)
        st.markdown("**Quantitative analysis of excavation progression and growth rates**")
        st.markdown("---")
        
        for img in categories["growth_metrics"]:
            path = os.path.join(out_dir, img)
            title = os.path.splitext(img)[0].replace("_", " ").title()
            caption = f"Metric: {img}"
            
            display_image_with_style(path, title, caption)
            st.markdown("---")
    
    # No-Go Violations Section
    if categories["no_go_violations"]:
        st.markdown('<div class="section-header">No-Go Zone Violation Analysis</div>', unsafe_allow_html=True)
        st.markdown("**Monitoring of restricted area violations and boundary compliance**")
        st.markdown("---")
        
        for img in categories["no_go_violations"]:
            path = os.path.join(out_dir, img)
            title = os.path.splitext(img)[0].replace("_", " ").title()
            caption = f"Violation Analysis: {img}"
            
            display_image_with_style(path, title, caption)
            st.markdown("---")
    
    # Other Analysis
    if categories["other"]:
        st.markdown('<div class="section-header"> Additional Analysis</div>', unsafe_allow_html=True)
        
        for img in categories["other"]:
            path = os.path.join(out_dir, img)
            title = os.path.splitext(img)[0].replace("_", " ").title()
            caption = f"File: {img}"
            
            display_image_with_style(path, title, caption)
            st.markdown("---")
    
    if not any([categories["area_analysis"], categories["growth_metrics"], 
                categories["no_go_violations"], categories["other"]]):
        st.info("No temporal analysis data available for this mine site.")

# TAB 3: DATA TABLES

with tab3:
    st.markdown('<div class="section-header"> Data Reports & Tables</div>', unsafe_allow_html=True)
    
    if not csvs:
        st.info("No data reports available for this mine site.")
    else:
        selected_csv = st.selectbox(
            " Select Report:",
            csvs,
            format_func=lambda x: "📄 " + x.replace("_", " ").replace(".csv", "").title(),
            help="Choose a data report to view detailed information"
        )
        
        csv_path = os.path.join(out_dir, selected_csv)
        
        try:
            df = pd.read_csv(csv_path)
            
            # Display metrics
            st.markdown("### Report Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(" Total Records", f"{len(df):,}")
            with col2:
                st.metric(" Data Fields", len(df.columns))
            '''with col3:
                st.metric(" File Size", pretty_file_size(csv_path))'''
            
            st.markdown("---")
            
            # Data preview
            st.markdown("### Report Data")
            # Reset index to start from 1
            df_display = df.copy()
            df_display.index = df_display.index + 1
            st.dataframe(df_display, use_container_width=True, height=500)
            
            # Actions
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col2:
                st.download_button(
                    label="⬇️ Download Report (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=selected_csv,
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                # Excel download option
                try:
                    excel_data = df.to_excel(index=False)
                    st.download_button(
                        label="⬇️ Download Report (Excel)",
                        data=excel_data,
                        file_name=selected_csv.replace('.csv', '.xlsx'),
                        mime="application/vnd.ms-excel",
                        use_container_width=True
                    )
                except:
                    pass
            
            # Statistical summary
            with st.expander("View Statistical Summary"):
                st.dataframe(df.describe(), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading report: {str(e)}")
            st.exception(e)

# TAB 4: ALERT SYSTEM

with tab4:
    st.markdown('<div class="section-header">Alert & Violation Management System</div>', unsafe_allow_html=True)
    
    if not os.path.exists(alert_log):
        st.success(" No alerts recorded. All mining operations are within permitted boundaries.")
        st.info("The system continuously monitors for violations and will generate alerts if any issues are detected.")
    else:
        alerts_df = parse_alerts(alert_log)
        
        if alerts_df is None or alerts_df.empty:
            st.success("Alert log exists but contains no active alerts.")
        else:
            # Alert summary metrics
            st.markdown("### Alert Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Alerts", len(alerts_df))
            
            with col2:
                if 'Severity' in alerts_df.columns:
                    critical = len(alerts_df[alerts_df['Severity'].isin(['CRITICAL', 'ERROR'])])
                    st.metric("Critical/Error", critical, 
                             delta=None if critical == 0 else f"{critical} active", 
                             delta_color="inverse")
            
            with col3:
                if 'Severity' in alerts_df.columns:
                    warnings = len(alerts_df[alerts_df['Severity'] == 'WARNING'])
                    st.metric("Warnings", warnings,
                             delta=None if warnings == 0 else f"{warnings} active",
                             delta_color="inverse")
            
            with col4:
                if 'Severity' in alerts_df.columns:
                    info = len(alerts_df[alerts_df['Severity'] == 'INFO'])
                    st.metric("Info", info)
            
            st.markdown("---")
            
            # Filters and controls
            st.markdown("### Filter & Sort Options")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if 'Severity' in alerts_df.columns:
                    all_severities = sorted(alerts_df['Severity'].unique())
                    severity_filter = st.multiselect(
                        "Filter by Severity Level:",
                        options=all_severities,
                        default=all_severities,
                        help="Select severity levels to display"
                    )
                    filtered_df = alerts_df[alerts_df['Severity'].isin(severity_filter)]
                else:
                    filtered_df = alerts_df
            
            with col2:
                sort_order = st.radio(
                    " Sort Order:",
                    ["Newest First", "Oldest First"],
                    horizontal=False
                )
            
            # Sort data
            if 'Timestamp' in filtered_df.columns:
                try:
                    filtered_df = filtered_df.copy()
                    # Convert to datetime for proper sorting
                    filtered_df['sort_date'] = pd.to_datetime(filtered_df['Timestamp'], errors='coerce')
                    # Sort by date
                    filtered_df = filtered_df.sort_values('sort_date', ascending=(sort_order == "Oldest First"))
                    # Drop the helper column
                    filtered_df = filtered_df.drop('sort_date', axis=1)
                    # Reset index to get proper serial numbers starting from 1
                    filtered_df = filtered_df.reset_index(drop=True)
                    filtered_df.index = filtered_df.index + 1
                except:
                    filtered_df = filtered_df.reset_index(drop=True)
                    filtered_df.index = filtered_df.index + 1
            
            st.markdown("---")
            
            # Display alerts with professional color coding
            def highlight_severity(row):
                if 'Severity' not in row:
                    return [''] * len(row)
                
                severity = str(row['Severity']).upper()
                if severity in ['CRITICAL', 'ERROR']:
                    return ['background-color: rgba(239, 68, 68, 0.15); color: #ef4444; font-weight: 600; border-left: 4px solid #ef4444;'] * len(row)
                elif severity == 'WARNING':
                    return ['background-color: rgba(251, 191, 36, 0.15); color: #fbbf24; font-weight: 500; border-left: 4px solid #fbbf24;'] * len(row)
                elif severity == 'INFO':
                    return ['background-color: rgba(59, 130, 246, 0.15); color: #3b82f6; border-left: 4px solid #3b82f6;'] * len(row)
                return [''] * len(row)
            
            st.markdown("### Alert Log Details")
            st.dataframe(
                filtered_df.style.apply(highlight_severity, axis=1),
                use_container_width=True,
                height=500
            )
            
            # Export options
            st.markdown("---")
            st.markdown("### Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label=" Export Filtered Alerts (CSV)",
                    data=filtered_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"mine_{mine_id}_alerts_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    label=" Export All Alerts (CSV)",
                    data=alerts_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"mine_{mine_id}_alerts_complete_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                with open(alert_log, 'r') as f:
                    raw_log = f.read()
                st.download_button(
                    label="Export Raw Log File",
                    data=raw_log,
                    file_name=f"mine_{mine_id}_alerts_raw_{datetime.now().strftime('%Y%m%d')}.log",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Raw log viewer
            with st.expander("View Raw Log File"):
                st.code(raw_log, language="text", line_numbers=True)

# FOOTER
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"""
**AURORA 2.0 - Satellite-Based Mining Monitoring System**  
Mine Site {mine_id} |  Data Location: `{out_dir}`  
🇮🇳 Developed for ISRO |  Confidential & Authorized Use Only  
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
st.markdown('</div>', unsafe_allow_html=True)
