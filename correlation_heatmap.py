import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# CONFIGURATION & STYLE
# ==========================================
CONFIG = {
    'input_file': 'btc_macro_dataset.csv',
    'output_dir': 'btc_study_visuals'
}

if not os.path.exists(CONFIG['output_dir']):
    os.makedirs(CONFIG['output_dir'])

try:
    plt.style.use('seaborn-v0_8-paper')
except:
    try:
        plt.style.use('seaborn-paper')
    except:
        plt.style.use('ggplot')

plt.rcParams.update({
    'font.family': 'serif',
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.autolayout': True
})

# ==========================================
# DATA LOADING
# ==========================================
def load_data_for_heatmap():
    print("--- Loading Data for Heatmap ---")
    if not os.path.exists(CONFIG['input_file']):
        raise FileNotFoundError(f"Input file {CONFIG['input_file']} not found.")

    df = pd.read_csv(CONFIG['input_file'], index_col=0, parse_dates=True)
    
    # 1. BTC Returns
    df['Ret'] = np.log(df['BTC_Price']).diff()
    
    # 2. Log Volatility (Rogers-Satchell priority)
    if 'BTC_RogersSatchell' in df.columns:
        print("  > Using Rogers-Satchell Volatility.")
        rs_clean = df['BTC_RogersSatchell'].clip(lower=1e-8).fillna(1e-8)
        df['Log_Vol'] = np.log(rs_clean)
    else:
        print("  > Rogers-Satchell not found. Using Squared Returns.")
        sq_ret = (df['Ret']**2).replace(0, 1e-8)
        df['Log_Vol'] = np.log(sq_ret)

    # 3. BTC Volume (Added)
    # Volume is often non-stationary, so we take log-diff
    if 'FUND_BTC_Volume' in df.columns:
        vol_clean = df['FUND_BTC_Volume'].replace(0, np.nan).ffill()
        df['Vol_Chg'] = np.log(vol_clean).diff()
    else:
        df['Vol_Chg'] = 0

    # 4. Create Clean DataFrame for Correlation
    data = pd.DataFrame({
        'BTC_Returns': df['Ret'],
        'Log_Volatility': df['Log_Vol'],
        'BTC_Volume_Chg': df['Vol_Chg'], # Added Volume
        'WIKI_Inflation_Chg': np.log(df['WIKI_Inflation']).diff(),
        'GT_Bitcoin_Chg': df['GT_Bitcoin'].diff(),
        'VIX_Chg': np.log(df['MKT_VIX']).diff()
    }).dropna()
    
    return data

# ==========================================
# PLOTTING ROUTINE
# ==========================================
def plot_correlation_heatmap(df):
    print("--- Generating Correlation Heatmap ---")
    
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap='coolwarm',
        vmax=0.5,
        vmin=-0.5,
        center=0,
        square=True, 
        linewidths=0.5, 
        cbar_kws={"shrink": 0.8},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 11},
        ax=ax
    )
    
    plt.title("Correlation Matrix (Inc. Volume)", fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right') 
    plt.yticks(rotation=0)
    
    output_path = os.path.join(CONFIG['output_dir'], 'Fig3_Correlation_Matrix.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  > Saved Heatmap to {output_path}")

if __name__ == "__main__":
    df = load_data_for_heatmap()
    plot_correlation_heatmap(df)
