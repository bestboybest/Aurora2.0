import os
import re
import glob
import streamlit as st
import pandas as pd

st.set_page_config(page_title="AURORA 2.0 UI", layout="wide")

MINE_DATA_DIR = "Mine Data"

#Helpers
def detect_all_mines():
    """
    Detect Mine IDs:
    """
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
    # CHANGED: Look in Outputs folder instead
    return os.path.join(outputs_dir(mid), f"mine_{mid}_alerts.log")

def list_files(folder: str):
    if not os.path.exists(folder):
        return []
    return sorted(os.listdir(folder))

def pretty_file_size(path):
    size = os.path.getsize(path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

def parse_alerts(log_path):
    """
    Parse alerts log file and return as DataFrame
    Expected format: timestamp | severity | message
    """
    if not os.path.exists(log_path):
        return None
    
    alerts = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Try to parse common log formats
                # Format 1: "2024-01-15 10:30:45 | WARNING | Message here"
                parts = line.split('|')
                if len(parts) >= 3:
                    alerts.append({
                        'Timestamp': parts[0].strip(),
                        'Severity': parts[1].strip(),
                        'Message': parts[2].strip()
                    })
                else:
                    # Format 2: Plain text, treat as INFO
                    alerts.append({
                        'Timestamp': 'N/A',
                        'Severity': 'INFO',
                        'Message': line
                    })
        
        return pd.DataFrame(alerts) if alerts else None
    
    except Exception as e:
        st.error(f"Error reading alerts: {e}")
        return None


#UI
st.title("AURORA 2.0: Precomputed Monitoring Dashboard")
st.write( """ This dashboard shows **pre-generated monitoring results** for each mine.  
Pick a mine and view its maps/plots/alerts. """)

available_mines = detect_all_mines()

if not available_mines:
    st.error("No Mine folders found inside.")
    st.stop()

#Sidebar
st.sidebar.header("Mine Selection")
mine_id = st.sidebar.selectbox("Choose Mine ID", available_mines)
st.sidebar.markdown("---")
st.sidebar.info("This UI displays results from Mine Data")

st.subheader(f"Selected Mine: {mine_id}")
st.code(mine_folder(mine_id))

#Load Outputs
out_dir = outputs_dir(mine_id)
alert_log = alerts_file(mine_id)

st.divider()
st.header("Outputs")

if not os.path.exists(out_dir):
    st.warning("Outputs folder not found for this mine.")
    st.info(f"Expected: `{out_dir}`")
    st.stop()

files = list_files(out_dir)

if not files:
    st.warning("Outputs folder exists but is empty.")
    st.stop()

images = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
csvs = [f for f in files if f.lower().endswith(".csv")]

tab1, tab2, tab3 = st.tabs(["Maps & Plots", "Tables / Alerts", "Alert Logs"])


#Images Tab
with tab1:
    if not images:
        st.info("No map/plot images found.")
    else:
        st.success(f"Found {len(images)} images")

        #SORT IMAGES FOR UI (ALL MINES)
        def sort_images_for_ui(images):
            suffix_order = [
                "spatialMap_0percent",
                "spatialMap_25percent",
                "spatialMap_50percent",
                "spatialMap_75percent",
                "spatialMap_100percent",
                "AreavsTime",
                "NormalizedExcavation",
                "GrowthRate",
                "CandidateAreavsTime",
                "ComparisionPlot",
                "excavationProgress",
                "FirstSeenPlot",
                "NoGo_Excavation_vs_Time",
            ]
            suffix_rank = {s: i for i, s in enumerate(suffix_order)}

            def get_suffix(fname):
                base = os.path.splitext(fname)[0]
                parts = base.split("_", 2)      # mine, id, suffix
                return parts[2] if len(parts) >= 3 else base

            return sorted(images, key=lambda x: suffix_rank.get(get_suffix(x), 10**9))

        images = sort_images_for_ui(images)

        #keep index valid after sorting
        if "img_idx" in st.session_state:
            st.session_state.img_idx %= len(images)

        #Keep current index across reruns
        if "img_idx" not in st.session_state:
            st.session_state.img_idx = 0

        #Column controls
        colA, colB, colC = st.columns([2, 8, 1])

        with colA:
            if st.button("⬅️ Previous", key="img_prev"):
                st.session_state.img_idx = (st.session_state.img_idx - 1) % len(images)

        with colC:
            if st.button("Next ➡️", key="img_next"):
                st.session_state.img_idx = (st.session_state.img_idx + 1) % len(images)

        #Show the selected image
        img = images[st.session_state.img_idx]
        path = os.path.join(out_dir, img)

        #heading
        st.markdown(f"### {os.path.splitext(img)[0]}")
        st.caption(
            f"Image {st.session_state.img_idx + 1} / {len(images)} | "
            f"Size: {pretty_file_size(path)}"
        )
        st.image(path, use_container_width=True)


#CSV Tab
with tab2:
    if not csvs:
        st.info("No CSV files found.")
    else:
        chosen = st.selectbox("Select CSV file", csvs)
        csv_path = os.path.join(out_dir, chosen)

        st.caption(f"File: `{chosen}` | Size: {pretty_file_size(csv_path)}")

        try:
            df = pd.read_csv(csv_path)
            st.dataframe(df, use_container_width=True)

            st.download_button(
                label="⬇️ Download this CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=chosen,
                mime="text/csv"
            )

        except Exception as e:
            st.error("Failed to read CSV.")
            st.exception(e)


#Alert Logs Tab
with tab3:
    st.subheader("Alert Logs")
    
    if not os.path.exists(alert_log):
        st.warning(f"Alert log file not found.")
        st.info("No alerts recorded for this mine.")
    else:
        # Show file info
        st.success(f"Alert log found | Size: {pretty_file_size(alert_log)}")
        
        # Parse and display alerts
        alerts_df = parse_alerts(alert_log)
        
        if alerts_df is None or alerts_df.empty:
            st.info("Alert log is empty or could not be parsed.")
        else:
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Alerts", len(alerts_df))
            with col2:
                if 'Severity' in alerts_df.columns:
                    warnings = len(alerts_df[alerts_df['Severity'].str.contains('WARNING', case=False, na=False)])
                    st.metric("Warnings", warnings)
            with col3:
                if 'Severity' in alerts_df.columns:
                    errors = len(alerts_df[alerts_df['Severity'].str.contains('ERROR|CRITICAL', case=False, na=False)])
                    st.metric("Errors/Critical", errors)
            
            st.markdown("---")
            
            # Filter options
            if 'Severity' in alerts_df.columns:
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    options=alerts_df['Severity'].unique(),
                    default=alerts_df['Severity'].unique()
                )
                
                filtered_df = alerts_df[alerts_df['Severity'].isin(severity_filter)]
            else:
                filtered_df = alerts_df
            
            # Display table with color coding
            def highlight_severity(row):
                if 'Severity' not in row:
                    return [''] * len(row)
                
                severity = str(row['Severity']).upper()
                if 'CRITICAL' in severity or 'ERROR' in severity:
                    return ['background-color: #ffcccc'] * len(row)
                elif 'WARNING' in severity:
                    return ['background-color: #fff3cd'] * len(row)
                elif 'INFO' in severity:
                    return ['background-color: #d1ecf1'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                filtered_df.style.apply(highlight_severity, axis=1),
                use_container_width=True,
                height=400
            )
            
            # Download button
            st.download_button(
                label="⬇️ Download Alert Log",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name=f"mine_{mine_id}_alerts.csv",
                mime="text/csv"
            )
            
            # Show raw log option
            with st.expander("📄 View Raw Log File"):
                try:
                    with open(alert_log, 'r') as f:
                        raw_content = f.read()
                    st.code(raw_content, language="text")
                except Exception as e:
                    st.error(f"Could not read raw file: {e}")
