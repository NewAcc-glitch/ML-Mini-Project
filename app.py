import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Ocean Data Dashboard", layout="wide")

# Title
st.title("🌊 Ocean Data Analysis Dashboard")
st.success("✅ App is successfully deployed!")

# Sample data section
st.header("📊 Sample Data Preview")

# Create sample ocean data
@st.cache_data
def load_sample_data():
    dates = pd.date_range('1993-01-01', '2023-12-31', freq='M')
    n_samples = len(dates)
    np.random.seed(42)
    
    df = pd.DataFrame({
        'time': dates,
        'SST': 20 + 5*np.sin(2*np.pi*np.arange(n_samples)/12) + 0.1*np.arange(n_samples) + np.random.normal(0, 1, n_samples),
        'Salinity': 35 + 0.5*np.cos(2*np.pi*np.arange(n_samples)/12) + np.random.normal(0, 0.2, n_samples),
        'MLD': 50 + 10*np.sin(2*np.pi*np.arange(n_samples)/6) + np.random.normal(0, 5, n_samples),
    })
    return df

df = load_sample_data()

# Show basic info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Records", len(df))
with col2:
    st.metric("Date Range", f"{df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}")
with col3:
    st.metric("Variables", len(df.columns) - 1)  # excluding time

# Data preview
st.subheader("Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# Basic visualization
st.subheader("📈 Basic Visualizations")

tab1, tab2, tab3 = st.tabs(["Time Series", "Distributions", "Correlations"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['time'], df['SST'], label='Sea Surface Temperature', color='red')
    ax.set_title("Sea Surface Temperature Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("SST (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df['SST'], kde=True, ax=ax, color='red')
        ax.set_title("SST Distribution")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df['Salinity'], kde=True, ax=ax, color='blue')
        ax.set_title("Salinity Distribution")
        st.pyplot(fig)

with tab3:
    # Calculate correlation matrix
    corr_matrix = df[['SST', 'Salinity', 'MLD']].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

# File uploader for future use
st.header("📁 Upload Your Own Data")
uploaded_file = st.file_uploader("Upload a CSV file with ocean data", type=['csv'])
if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! {len(user_df)} records loaded.")
        st.dataframe(user_df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

# Footer
st.markdown("---")
st.markdown("### 🚀 Next Steps")
st.markdown("""
1. **Test this deployment** - If you see this, your app is working!
2. **Add your EDA code** - Gradually add back your complex visualizations
3. **Upload real data** - Use the file uploader above
4. **Customize** - Add your specific ocean data analysis
""")
