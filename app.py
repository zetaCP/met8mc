import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os

st.set_page_config(layout="wide", page_title="Neuron Dashboard", page_icon="🧠")

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
scale_type = 'gOSI'
master_order = ['L2a', 'L2b', 'L2c', 'L3a', 'L3b', 'L4a', 'L4b', 'L4c', 'L5a', 'L5b','L5NP', 'L6a', 'L6b','L6short-a',
'DTC','ITC', 'PTC','STC'] 
cmap = plt.get_cmap('tab20')
color_map = {cat: mcolors.to_hex(cmap(i)) for i, cat in enumerate(master_order)}

# Automatically find any CSV files matching the naming scheme in the current directory!
csv_files = glob.glob("*_digital_twin_PrePC_SST_PostPC_reciprocal_synapses_count.csv")
neuron_ids = [f.replace("_digital_twin_PrePC_SST_PostPC_reciprocal_synapses_count.csv", "") for f in csv_files]

if not neuron_ids:
    st.error(f"No *_reciprocal_synapses_count.csv files found in {os.getcwd()}!")
    st.stop()

# ==========================================
# 2. DATA LOADING (Using Streamlit Cache for speed!)
# ==========================================
@st.cache_data
def load_data_for_neuron(cell_name):
    file_path = f"{cell_name}_digital_twin_PrePC_SST_PostPC_reciprocal_synapses_count.csv"
    try:
        df = pd.read_csv(file_path).dropna(subset=['readout_loc_x', 'readout_loc_y', 'pref_ori', 'cell_type', 'gOSI', 'cc_abs', 'horizontal_dist']).copy()
    except FileNotFoundError:
        return pd.DataFrame() 
    
    if len(df) > 0:
        min_len, max_len = 0.005, 0.02
        min_v, max_v = df[scale_type].min(), df[scale_type].max()
        df['arrow_length'] = min_len if max_v == min_v else min_len + (df[scale_type] - min_v) * (max_len - min_len) / (max_v - min_v)
        theta = np.deg2rad(df['pref_ori'])
        df['u'], df['v'] = df['arrow_length'] * np.cos(theta), df['arrow_length'] * np.sin(theta)
        
        # Add a unique ID for Plotly click detection
        df['unique_click_id'] = df.index.astype(str)
    return df

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🧠 Cell Filters")

selected_neuron = st.sidebar.selectbox("Select Neuron ID:", neuron_ids)
df = load_data_for_neuron(selected_neuron).copy()

if len(df) == 0:
    st.warning("Selected cell has no valid data or file is missing.")
    st.stop()

st.sidebar.markdown("---")

w_gOSI = st.sidebar.slider("Min gOSI:", float(df['gOSI'].min()), float(max(df['gOSI'].min(), df['gOSI'].max())), float(df['gOSI'].min()), step=0.05)
w_cc_abs = st.sidebar.slider("Min cc_abs:", float(df['cc_abs'].min()), float(max(df['cc_abs'].min(), df['cc_abs'].max())), float(df['cc_abs'].min()), step=0.05)

n_pre_min, n_pre_max = int(df['n_pre'].min()), int(max(df['n_pre'].min(), df['n_pre'].max()))
w_n_pre = st.sidebar.slider("Min n_pre:", n_pre_min, n_pre_max, n_pre_min, step=1)

n_post_min, n_post_max = int(df['n_post'].min()), int(max(df['n_post'].min(), df['n_post'].max()))
w_n_post = st.sidebar.slider("Min n_post:", n_post_min, n_post_max, n_post_min, step=1)

h_min, h_max = float(df['horizontal_dist'].min()), float(max(df['horizontal_dist'].min(), df['horizontal_dist'].max()))
w_horiz_dist = st.sidebar.slider("Max horiz Dist:", h_min, h_max, h_max, step=10.0)

# Manual Proof Dropdown
if 'manual_proof_by_pat' in df.columns:
    w_manual_proof = st.sidebar.selectbox("Manual Proof By Pat:", ["All", True, False])
else:
    w_manual_proof = "All"

# ==========================================
# 4. FILTERING & GRAPH GENERATION
# ==========================================
mask = (df['gOSI'] >= w_gOSI) & \
       (df['cc_abs'] >= w_cc_abs) & \
       (df['n_pre'] >= w_n_pre) & \
       (df['n_post'] >= w_n_post) & \
       (df['horizontal_dist'] <= w_horiz_dist)

if 'manual_proof_by_pat' in df.columns and w_manual_proof != "All":
    mask = mask & (df['manual_proof_by_pat'] == w_manual_proof)

df_filt = df[mask].reset_index(drop=True)

col1, col2 = st.columns([3, 1])

with col1:
    st.title(f"{selected_neuron[4:] if len(selected_neuron) > 4 else selected_neuron} (N={len(df_filt)})")
    st.markdown(f"**Arrow size determined by `{scale_type}`**")

    fig_widget = go.Figure()

    for cat in master_order:
        df_cat = df_filt[df_filt['cell_type'] == cat]
        if len(df_cat) > 0:
            q = ff.create_quiver(
                x=df_cat['readout_loc_x'].tolist(), y=df_cat['readout_loc_y'].tolist(), 
                u=df_cat['u'].tolist(), v=df_cat['v'].tolist(), 
                scale=1, arrow_scale=0.35, angle=np.pi/12, 
                name=cat, line=dict(color=color_map[cat], width=2.0)
            )
            # Disable hover on lines to make clicking the center cleaner
            q.data[0].hoverinfo = 'skip'
            fig_widget.add_trace(q.data[0])
            
            hover_texts = ["<br>".join([f"<b>{k}</b>: {v}" for k, v in row.items() if k not in ['u','v','arrow_length', 'unique_click_id']]) for _, row in df_cat.iterrows()]
            
            # Invisible scatter points for click detection!
            fig_widget.add_trace(go.Scatter(
                x=df_cat['readout_loc_x'].tolist(), y=df_cat['readout_loc_y'].tolist(), 
                mode='markers', marker=dict(size=12, color='rgba(0,0,0,0)'), 
                showlegend=False, hoverinfo='text', text=hover_texts,
                customdata=df_cat['unique_click_id'].tolist() 
            ))

    pad_x = 1 if df['readout_loc_x'].max() == df['readout_loc_x'].min() else (df['readout_loc_x'].max() - df['readout_loc_x'].min()) * 0.05
    pad_y = 1 if df['readout_loc_y'].max() == df['readout_loc_y'].min() else (df['readout_loc_y'].max() - df['readout_loc_y'].min()) * 0.05

    fig_widget.update_layout(
        width=900, height=800, template="plotly_white",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        xaxis=dict(range=[df['readout_loc_x'].min() - pad_x, df['readout_loc_x'].max() + pad_x], autorange=False),
        yaxis=dict(range=[df['readout_loc_y'].min() - pad_y, df['readout_loc_y'].max() + pad_y], autorange=False, scaleanchor="x", scaleratio=1), 
        legend_title_text="Cell Type"
    )

    # Render Chart with Streamlit 1.35+ Custom On-Select behaviour!
    event = st.plotly_chart(fig_widget, on_select="rerun", selection_mode="points", use_container_width=True)

# Process click events natively in Streamlit for our text box
with col2:
    st.markdown("### 📋 Selected Details")
    clicked_data = event.selection.points if getattr(event, 'selection', None) else []
    
    if clicked_data:
        selected_id = clicked_data[0].get("customdata")
        if selected_id:
            row = df.loc[df['unique_click_id'] == str(selected_id)].iloc[0]
            clean_str = "\n".join([f"{k}: {v}" for k, v in row.items() if k not in ['u','v','arrow_length', 'unique_click_id']])
            st.code(clean_str, language='yaml') # Beautiful, selectable code block!
    else:
        st.info("Click on any point in the center of an arrow on the graph to see its detailed values here!")
