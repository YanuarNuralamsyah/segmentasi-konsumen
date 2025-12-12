import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # NEW: Library for 3D Plotting
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Retail Customer Segmentation (SOM)",
    page_icon="🛍️",
    layout="wide"
)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_data(uploaded_file):
    """
    Loads data from a CSV file. If no file is provided, 
    it generates synthetic data based on the 'Mall Customers' distribution.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset Uploaded Successfully!")
        return df
    else:
        # Fallback: Synthetic Data Generation
        st.warning("⚠️ No CSV uploaded. Using Synthetic Demo Data.")
        np.random.seed(42)
        data = {
            'CustomerID': range(1, 101),
            'Age': np.random.randint(18, 70, 100),
            'Annual Income (k$)': np.random.randint(15, 137, 100),
            'Spending Score (1-100)': np.random.randint(1, 100, 100)
        }
        return pd.DataFrame(data)

# ==========================================
# 3. SIDEBAR: HYPERPARAMETERS
# ==========================================
st.sidebar.header("⚙️ Model Hyperparameters")
st.sidebar.markdown("Adjust the Neural Network configuration here.")

map_size = st.sidebar.slider("Map Grid Size (N x N)", min_value=5, max_value=50, value=10, help="Menentukan jumlah total neuron (klaster). Grid 10x10 berarti ada 100 neuron. Semakin besar grid, semakin detail pemetaannya namun komputasi lebih berat.")
sigma_val = st.sidebar.slider("Sigma (Radius)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, help="Radius tetangga yang ikut diperbarui saat ada pemenang. Sigma besar bagus di awal untuk pengelompokan global, sigma kecil untuk detail lokal.")
lr_val = st.sidebar.slider("Learning Rate", min_value=0.1, max_value=2.0, value=0.5, step=0.1, help="Seberapa drastis bobot neuron berubah setiap iterasi. Nilai besar mempercepat belajar, nilai kecil memperhalus (stabilisasi).")
epochs = st.sidebar.number_input("Iterations (Epochs)", min_value=100, max_value=100000, value=1000, step=100, help="Berapa kali model melihat seluruh data. Iterasi yang lebih banyak biasanya menghasilkan peta yang lebih stabil (konvergen).")

# ==========================================
# 4. MAIN APPLICATION LOGIC
# ==========================================
st.title("🛍️ Retail Customer Segmentation App")
st.markdown("""
**Methodology:** This application utilizes the **Self-Organizing Maps (SOM)** algorithm 
to cluster customers based on shopping behavior.
""")

# --- Step A: Data Ingestion ---
st.header("1. Data Ingestion")
uploaded_file = st.file_uploader("Upload 'Mall_Customers.csv'", type=['csv'])
df = load_data(uploaded_file)

with st.expander("Show Raw Data Preview"):
    st.dataframe(df.head())

# --- Step B: Preprocessing ---
st.header("2. Preprocessing & Feature Selection")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'CustomerID' in numeric_cols: numeric_cols.remove('CustomerID')

# User selection for clustering features
selected_features = st.multiselect(
    "Select Features for Clustering:", 
    options=numeric_cols, 
    default=[c for c in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'] if c in numeric_cols],
    help="Pilih variabel yang menentukan kemiripan pelanggan. Data ini akan otomatis dinormalisasi (MinMax Scaling) sebelum diproses."
)

if len(selected_features) < 2:
    st.error("Please select at least 2 features to proceed.")
    st.stop()

# Scaling Data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[selected_features])
st.caption(f"Data Scaled to [0-1] range. Input Shape: {data_scaled.shape}")

# --- Step C: Training & Visualization ---
st.header("3. SOM Training & Analysis")

if st.button("🚀 Train Neural Network"):
    with st.spinner("Training SOM... Please wait."):
        
        # 1. Initialize and Train SOM
        som = MiniSom(x=map_size, y=map_size, input_len=len(selected_features), 
                      sigma=sigma_val, learning_rate=lr_val)
        som.random_weights_init(data_scaled)
        som.train_random(data_scaled, num_iteration=epochs)
        
        # 2. Calculate Metrics (HITUNG DUA-DUANYA)
        qe = som.quantization_error(data_scaled)
        te = som.topographic_error(data_scaled)
        
        st.success("Training Complete!")
        
        # Tampilkan dalam kolom metrik yang rapi
        m1, m2 = st.columns(2)
        m1.metric("Quantization Error (QE)", f"{qe:.4f}", help="Rata-rata jarak data ke neuron. Makin kecil makin pas.")
        m2.metric("Topographic Error (TE)", f"{te:.4f}", help="Rasio data yang neuron pemenangnya tidak bertetangga. Makin kecil makin rapi.")

        # 3. Get Cluster IDs for every customer
        # Map each data point to its winning neuron
        winner_coordinates = np.array([som.winner(x) for x in data_scaled]).T
        cluster_index = np.ravel_multi_index(winner_coordinates, (map_size, map_size))
        
        # Add Cluster ID to original dataframe for visualization
        df['Cluster_ID'] = cluster_index
        df['Cluster_ID'] = df['Cluster_ID'].astype(str) # Convert to string for discrete coloring

    # --- Step D: 2D Visualizations ---
    st.subheader("Visualizing the Neural Network")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**A. U-Matrix (Distance Map)**")
        st.caption("Dark areas = Dense Clusters. Light areas = Boundaries.")
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        u_matrix = som.distance_map()
        img = ax1.imshow(u_matrix, cmap='bone_r') 
        plt.colorbar(img, ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.markdown("**B. SOM Grid Distribution**")
        st.caption("Customers mapped to neurons (with jitter).")
        # Jitter logic
        jitter_x = (np.random.rand(len(data_scaled)) - 0.5) * 0.6
        jitter_y = (np.random.rand(len(data_scaled)) - 0.5) * 0.6
        
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        scatter = ax2.scatter(
            winner_coordinates[0] + jitter_x, 
            winner_coordinates[1] + jitter_y, 
            c=cluster_index, cmap='viridis', s=50, alpha=0.8, edgecolors='w'
        )
        st.pyplot(fig2)

    # --- Step E: 3D Visualization (NEW FEATURE) ---
    st.header("4. Interactive 3D Clustering")
    st.markdown("Visualize how the SOM clusters relate to the original customer data (Age, Income, Spending Score).")

    # Check if we have at least 3 features for a 3D plot
    if len(selected_features) >= 3:
        # Create 3D Scatter Plot using Plotly
        fig_3d = px.scatter_3d(
            df, 
            x=selected_features[0], 
            y=selected_features[1], 
            z=selected_features[2],
            color='Cluster_ID',
            title=f"3D Cluster View: {selected_features[0]} vs {selected_features[1]} vs {selected_features[2]}",
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.warning("⚠️ Please select at least 3 features to generate a 3D Scatter Plot.")

    # --- Step F: Business Interpretation ---
    st.header("5. Segmentation Analysis")
    
    # Calculate Mean values for each cluster
    cluster_profile = df.groupby('Cluster_ID')[selected_features].mean()
    
    st.dataframe(cluster_profile.style.highlight_max(axis=0, color='lightgreen'))
    st.info("Tip: You can rotate the 3D plot above to find the 'Target' cluster (High Income, High Spending).")

else:
    st.info("Adjust parameters in the sidebar and click 'Train Neural Network' to start.")