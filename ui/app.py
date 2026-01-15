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

tab1, tab2 = st.tabs(["Maps & Plots", "Tables / Alerts"])


#Images Tab
with tab1:
    if not images:
        st.info("No map/plot images found.")
    else:
        st.success(f"Found {len(images)} images")

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

        st.markdown(f"### {img}")
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
            st.dataframe(df, width="stretch")

            st.download_button(
                label="⬇️ Download this CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=chosen,
                mime="text/csv"
            )

        except Exception as e:
            st.error("Failed to read CSV.")
            st.exception(e)
