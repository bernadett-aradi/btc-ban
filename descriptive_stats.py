import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'input_file': 'btc_macro_dataset.csv',
    'output_dir': 'btc_study_visuals',
    'break_date': '2021-05-21'
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
    'figure.autolayout': True,
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

def calculate_hurst(series):
    ts = series.values
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    try:
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]*2.0 
    except:
        return 0.5

# ==========================================
# 1. LOAD AND PREP
# ==========================================
def load_data():
    print("--- Loading Data ---")
    if not os.path.exists(CONFIG['input_file']):
        raise FileNotFoundError(f"Input file {CONFIG['input_file']} not found.")

    df = pd.read_csv(CONFIG['input_file'], index_col=0, parse_dates=True)
    df['Ret'] = np.log(df['BTC_Price']).diff()
    
    if 'BTC_RogersSatchell' in df.columns:
        print("  > Detected Rogers-Satchell Estimator.")
        rs_clean = df['BTC_RogersSatchell'].clip(lower=1e-8).fillna(1e-8)
        df['Log_Vol'] = np.log(rs_clean)
        df['Vol_Proxy'] = np.sqrt(rs_clean) 
    else:
        sq_ret = (df['Ret']**2).replace(0, 1e-8)
        df['Log_Vol'] = np.log(sq_ret)
        df['Vol_Proxy'] = df['Ret'].abs()
        
    # Prepare Volume Change
    if 'FUND_BTC_Volume' in df.columns:
        v = df['FUND_BTC_Volume'].replace(0, np.nan).ffill()
        df['Vol_Chg'] = np.log(v).diff()
    else:
        df['Vol_Chg'] = 0
    
    data = pd.DataFrame({
        'BTC_Ret': df['Ret'],
        'Log_Vol': df['Log_Vol'],
        'Vol_Proxy': df['Vol_Proxy'],
        'GT_Bit_Chg': df['GT_Bitcoin'].diff(),
        'BTC_Vol_Chg': df['Vol_Chg'], # Added
        'Wiki_Inf_Chg': np.log(df['WIKI_Inflation']).diff(),
        'VIX_Chg': np.log(df['MKT_VIX']).diff()
    }).dropna()
    
    return data

# ==========================================
# 2. EXTENDED STATISTICS
# ==========================================
def generate_extended_stats(df):
    print("--- Generating Extended Statistics ---")
    stats_list = []
    
    for col in df.columns:
        s = df[col]
        mean = s.mean()
        std = s.std()
        skew = stats.skew(s)
        kurt = stats.kurtosis(s)
        
        try: jb_stat, jb_p = stats.jarque_bera(s)
        except: jb_p = np.nan
            
        try: adf_stat, adf_p, _, _, _, _ = adfuller(s, autolag='AIC')
        except: adf_p = np.nan
        
        try:
            lb_res = acorr_ljungbox(s, lags=[10], return_df=True)
            if 'lb_pvalue' in lb_res.columns: q_p = lb_res['lb_pvalue'].iloc[0]
            else: q_p = lb_res.iloc[0, 1] 
        except: q_p = np.nan
        
        try: lm_stat, lm_p, _, _ = het_arch(s)
        except: lm_p = np.nan
            
        hurst = calculate_hurst(s)

        stats_list.append({
            'Variable': col.replace('_', ' '),
            'Mean': f"{mean:.4f}",
            'Std': f"{std:.4f}",
            'Skew': f"{skew:.2f}",
            'Kurt': f"{kurt:.2f}",
            'JB (p)': f"{jb_p:.3f}",
            'ADF (p)': f"{adf_p:.4f}",
            'LB Q(10) (p)': f"{q_p:.3f}",
            'ARCH (p)': f"{lm_p:.3f}",
            'Hurst': f"{hurst:.2f}"
        })
        
    res_df = pd.DataFrame(stats_list)
    latex_file = os.path.join(CONFIG['output_dir'], 'table_stats_extended.tex')
    res_df.to_latex(latex_file, index=False, caption="Descriptive Statistics")
    print(f"  > Saved Extended Stats to {latex_file}")
    return res_df

# ==========================================
# 3. VISUALS (Added Volume to Fig 2)
# ==========================================
def plot_regime_comparison_fixed(raw_df):
    print("--- Plotting Regime Visuals (Fixed Annotation) ---")
    plot_data = raw_df.loc['2019-01-01':'2023-12-31'].copy()
    
    def normalize(s): return (s - s.min()) / (s.max() - s.min())

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    window = 30
    trend_hype = normalize(plot_data['GT_Bitcoin'].rolling(window).mean())
    trend_macro = normalize(plot_data['WIKI_Inflation'].rolling(window).mean())
    
    # Add Volume Trend
    if 'FUND_BTC_Volume' in plot_data.columns:
        trend_vol = normalize(plot_data['FUND_BTC_Volume'].rolling(window).mean())
        l3 = ax1.plot(plot_data.index, trend_vol, color='#9467bd', label='Trading Activity (Volume)', alpha=0.6, lw=1.5, linestyle=':')
    else:
        l3 = []
    
    l1 = ax1.plot(plot_data.index, trend_hype, color='#ff7f0e', label='Retail Hype (Google: Bitcoin)', alpha=0.8, lw=2)
    l2 = ax1.plot(plot_data.index, trend_macro, color='#2ca02c', label='Macro Fear (Wiki: Inflation)', alpha=0.8, lw=2)
    
    break_dt = pd.to_datetime(CONFIG['break_date'])
    ax1.axvline(break_dt, color='black', linestyle='--', linewidth=2)
    
    trans = ax1.get_xaxis_transform()
    ax1.text(break_dt, 0.05, '  China Ban (May 2021)', transform=trans, 
             ha='left', va='bottom', fontweight='bold', fontsize=11, 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    ax1.set_ylabel("Normalized Index (30-day MA)")
    ax1.set_title("The Great Decoupling: Retail vs Macro vs Volume", fontweight='bold')
    
    lines = l1 + l2 + (l3 if l3 else [])
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(CONFIG['output_dir'], 'Fig2_Descriptive_Trends.pdf'))
    plt.close()

def plot_stylized_facts(df):
    print("--- Plotting Stylized Facts Panel ---")
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df['BTC_Ret'], kde=True, ax=ax1, color='blue', stat='density', label='Actual')
    try:
        mu, std = stats.norm.fit(df['BTC_Ret'])
        x = np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], 100)
        ax1.plot(x, stats.norm.pdf(x, mu, std), 'k--', label='Normal Dist')
    except: pass
    ax1.set_title("Panel A: Return Distribution", fontweight='bold')
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df.index, df['BTC_Ret'], color='gray', alpha=0.6, lw=0.5)
    ax2.set_title("Panel B: Volatility Clustering", fontweight='bold')
    ax2.set_ylim(-0.2, 0.2)
    
    ax3 = fig.add_subplot(gs[1, 0])
    sm.graphics.tsa.plot_acf(df['BTC_Ret'], lags=40, ax=ax3, zero=False, auto_ylims=True)
    ax3.set_title("Panel C: ACF of Returns", fontweight='bold')
    
    ax4 = fig.add_subplot(gs[1, 1])
    sm.graphics.tsa.plot_acf(df['Vol_Proxy'], lags=40, ax=ax4, zero=False, auto_ylims=True)
    ax4.set_title("Panel D: ACF of Volatility (Rogers-Satchell)", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'Fig4_Stylized_Facts.pdf'))
    plt.close()

if __name__ == "__main__":
    data = load_data()
    generate_extended_stats(data)
    raw_df = pd.read_csv(CONFIG['input_file'], index_col=0, parse_dates=True)
    plot_regime_comparison_fixed(raw_df)
    plot_stylized_facts(data)
