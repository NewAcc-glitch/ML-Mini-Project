# --- ENHANCED VISUALIZATION SECTION: Comprehensive EDA ---

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

# SUPPRESS FUTURE WARNINGS
warnings.filterwarnings('ignore', category=FutureWarning)

# Set better visual style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
print("STARTING COMPREHENSIVE EDA...")
# --- ENHANCED VISUALIZATION SECTION: Comprehensive EDA ---

# --- DATA LOADING & PREPARATION ---
# Try to load data if df doesn't exist
try:
    if 'df' not in locals() or df is None:
        print("Loading data...")
        # Add your actual data loading code here. Examples:
        # df = pd.read_csv('/kaggle/input/your-dataset/data.csv')
        # df = pd.read_csv('ocean_data.csv')
        
        # For demonstration, creating sample data if real data isn't available
        print("Using sample data - REPLACE WITH YOUR ACTUAL DATA LOADING")
        dates = pd.date_range('1993-01-01', '2017-12-31', freq='M')
        n_samples = len(dates)
        np.random.seed(42)
        
        df = pd.DataFrame({
            'time': dates,
            'SST': 20 + 5*np.sin(2*np.pi*np.arange(n_samples)/12) + 0.1*np.arange(n_samples) + np.random.normal(0, 1, n_samples),
            'Salinity': 35 + 0.5*np.cos(2*np.pi*np.arange(n_samples)/12) + np.random.normal(0, 0.2, n_samples),
            'U_current': np.random.normal(0, 0.5, n_samples),
            'V_current': np.random.normal(0, 0.3, n_samples),
            'MLD': 50 + 10*np.sin(2*np.pi*np.arange(n_samples)/6) + np.random.normal(0, 5, n_samples),
            'SSH': 0.1 + 0.05*np.sin(2*np.pi*np.arange(n_samples)/12) + np.random.normal(0, 0.02, n_samples)
        })
        
    # Data preparation
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Create enhanced time features
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)  # Cyclical encoding
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['current_magnitude'] = np.sqrt(df['U_current']**2 + df['V_current']**2)

    print(f"Data loaded: {len(df)} records from {df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}")

except Exception as e:
    print(f"Error loading data: {e}")
    print("Please make sure your dataframe 'df' is properly defined with columns: time, SST, Salinity, U_current, V_current, MLD, SSH")
    raise

# --- 1. MISSING VALUES ANALYSIS ---
print("\nMISSING VALUES ANALYSIS:")
print("Count:\n", df.isna().sum())
missing_percent = (df.isna().sum() / len(df)) * 100
print(f"\nPercentage:\n{missing_percent.round(2)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
sns.heatmap(df.isna(), yticklabels=False, cbar=True, cmap='viridis', ax=ax1)
ax1.set_title('Missing Values Pattern')

# Plot missing values percentage
missing_data = missing_percent[missing_percent > 0]
if len(missing_data) > 0:
    missing_data.plot(kind='bar', ax=ax2, color='coral')
    ax2.set_title('Columns with Missing Values (%)')
else:
    ax2.text(0.5, 0.5, 'No missing values!', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_title('Missing Values Check')

ax2.set_ylabel('Percentage')
plt.tight_layout()
plt.show()

# --- 2. MULTI-PANEL TIME SERIES ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
variables = ["SST", "Salinity", "MLD"]
colors = ['red', 'blue', 'green']

for idx, (var, color) in enumerate(zip(variables, colors)):
    if var in df.columns:
        axes[idx].plot(df["time"], df[var], color=color, linewidth=1, alpha=0.7, label=var)
        axes[idx].plot(df["time"], df[var].rolling(12, min_periods=1).mean(), color='black', 
                       linewidth=2, label='12M Rolling Mean')
        axes[idx].set_ylabel(var)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    else:
        axes[idx].text(0.5, 0.5, f'Column "{var}" not found', ha='center', va='center', transform=axes[idx].transAxes)

axes[-1].set_xlabel("Time")
plt.suptitle("Core Ocean Variables with Trend Analysis", fontsize=14, y=0.95)
plt.tight_layout()
plt.show()

# --- 3. ENHANCED CORRELATION ANALYSIS ---
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    corr_matrix = df[numeric_cols].corr()

    # Mask upper triangle for cleaner visualization
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="RdBu_r", center=0,
                square=True, fmt=".2f", cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix (Lower Triangle) - All Numeric Features")
    plt.tight_layout()
    plt.show()

    # Show strongest correlations
    corr_pairs = corr_matrix.unstack().sort_values(key=abs, ascending=False)
    print("Top 10 Strongest Correlations:")
    print(corr_pairs[corr_pairs != 1.0].head(10))
else:
    print("No numeric columns found for correlation analysis")

# --- 4. SMART PAIRPLOT (Sampled for Performance) ---
plot_vars = ["SST", "Salinity", "U_current", "V_current", "MLD"]
# Only use columns that exist in the dataframe
available_vars = [var for var in plot_vars if var in df.columns]

if len(available_vars) >= 2:
    sample_size = min(1000, len(df))
    sample_df = df[available_vars].sample(sample_size, random_state=42)

    print(f"\n📊 Generating pairplot with {sample_size} samples...")
    g = sns.PairGrid(sample_df, diag_sharey=False)
    g.map_upper(sns.scatterplot, alpha=0.6, s=15)
    g.map_lower(sns.kdeplot, fill=True, cmap="Blues", alpha=0.6)
    g.map_diag(sns.histplot, kde=True)
    plt.suptitle(f"Pairwise Relationships (Sampled: n={sample_size})", y=1.02)
    plt.show()
else:
    print(f"Not enough columns available for pairplot. Need at least 2, found {len(available_vars)}")

# --- 5. SEASONAL ANALYSIS ---
if 'SST' in df.columns and 'month' in df.columns:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Monthly distribution with enhanced styling
    sns.boxplot(x="month", y="SST", data=df, ax=ax1, palette="viridis")
    ax1.set_title("Seasonal Pattern: SST by Month")
    ax1.set_xlabel("Month")

    # Monthly trends with confidence intervals
    monthly_stats = df.groupby('month')['SST'].agg(['mean', 'std']).reset_index()
    ax2.plot(monthly_stats['month'], monthly_stats['mean'], marker='o', linewidth=2, 
             label='Monthly Mean', color='red')
    ax2.fill_between(monthly_stats['month'], 
                    monthly_stats['mean'] - monthly_stats['std'],
                    monthly_stats['mean'] + monthly_stats['std'],
                    alpha=0.3, label='±1 Std Dev', color='red')
    ax2.set_title("SST Monthly Pattern with Variability")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("SST")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- 6. DISTRIBUTION ANALYSIS ---
key_vars = ["SST", "Salinity", "U_current", "V_current", "MLD", "current_magnitude"]
available_key_vars = [var for var in key_vars if var in df.columns]

if available_key_vars:
    n_vars = len(available_key_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, var in enumerate(available_key_vars):
        sns.histplot(df[var], kde=True, ax=axes[idx])
        axes[idx].set_title(f'Distribution: {var}')
        # Add statistics annotation
        mean_val = df[var].mean()
        std_val = df[var].std()
        axes[idx].axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                          label=f'Mean: {mean_val:.2f}')
        axes[idx].legend()

    # Hide empty subplots
    for idx in range(len(available_key_vars), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle("Feature Distributions with Summary Statistics", y=0.98)
    plt.tight_layout()
    plt.show()

# --- 7. ENHANCED CURRENT ANALYSIS ---
if all(col in df.columns for col in ['U_current', 'V_current']):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Clean vector plot with sampling
    sample_size_currents = min(300, len(df))
    sample_idx = np.random.choice(len(df), sample_size_currents, replace=False)

    ax1.quiver(df.iloc[sample_idx]['U_current'], 
               df.iloc[sample_idx]['V_current'],
               scale=25, width=0.003, alpha=0.7, color='blue')
    ax1.set_title(f"Current Vectors (Sample: {sample_size_currents} points)")
    ax1.set_xlabel("U_current (eastward)")
    ax1.set_ylabel("V_current (northward)")
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Current magnitude distribution
    sns.histplot(df['current_magnitude'], kde=True, ax=ax2, color='green')
    ax2.set_title("Distribution of Current Magnitude")
    ax2.set_xlabel("Current Magnitude")
    ax2.axvline(df['current_magnitude'].mean(), color='red', linestyle='--',
               label=f'Mean: {df["current_magnitude"].mean():.3f}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"📈 Current Statistics - Magnitude: mean={df['current_magnitude'].mean():.4f}, "
          f"std={df['current_magnitude'].std():.4f}")

# --- 8. OUTLIER DETECTION ---
print("\nOUTLIER DETECTION (IQR Method):")
numeric_vars = ["SST", "Salinity", "MLD", "current_magnitude"]
available_numeric_vars = [var for var in numeric_vars if var in df.columns]

outlier_summary = {}
for var in available_numeric_vars:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)]
    outlier_percent = (len(outliers) / len(df)) * 100
    outlier_summary[var] = outlier_percent
    print(f"{var}: {len(outliers):>3} outliers ({outlier_percent:5.2f}%)")

# --- SUMMARY STATISTICS ---
print("\n" + "="*50)
print("EDA SUMMARY")
print("="*50)
print(f"Dataset period: {df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}")
print(f"Total records: {len(df):,}")
print(f"Features analyzed: {len(numeric_cols)} numeric columns")
print(f"Missing data: {df.isna().sum().sum()} total missing values")
print(f"Columns available: {list(df.columns)}")
print("COMPREHENSIVE EDA COMPLETE - Ready for feature engineering!")
