import os
import sys
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
from tqdm import tqdm

# ==========================================
# 0. CONFIGURATION & STYLE
# ==========================================
warnings.filterwarnings("ignore")

try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('seaborn-paper')

plt.rcParams.update({
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 1.5,
    'legend.fontsize': 10,
    'figure.autolayout': True
})

CONFIG = {
    'input_file': 'btc_macro_dataset.csv',
    'window_size': 730,       # Main Rolling Window (2 Years)
    'output_dir': 'btc_study_visuals',
    'break_date': '2021-05-21' # China Mining Crackdown
}

if not os.path.exists(CONFIG['output_dir']):
    os.makedirs(CONFIG['output_dir'])

# ==========================================
# 1. DATA PREPARATION
# ==========================================
def prep_data():
    print("--- 1. Preparing Data ---")
    if not os.path.exists(CONFIG['input_file']):
        raise FileNotFoundError("Input file not found. Run data_btc.py first.")

    print(f"  > Loading {CONFIG['input_file']}...")
    df = pd.read_csv(CONFIG['input_file'], index_col=0, parse_dates=True)
    
    # Returns (Log Returns)
    df['Ret'] = np.log(df['BTC_Price']).diff()
    
    # ------------------------------------------------------------------
    # CHANGE: Using Rogers-Satchell Estimator instead of Ret**2
    # ------------------------------------------------------------------
    if 'BTC_RogersSatchell' in df.columns:
        print("  > Using Rogers-Satchell Volatility Estimator.")
        # Ensure no negative values (numerical noise) and fill NaNs
        rs_vol = df['BTC_RogersSatchell'].clip(lower=0).fillna(0)
        
        df['Var_Total'] = rs_vol
        
        # For semi-variance (Down/Up), we use the direction of returns 
        # applied to the Rogers-Satchell magnitude
        df['Var_Down'] = np.where(df['Ret'] < 0, rs_vol, 0)
        df['Var_Up'] = np.where(df['Ret'] > 0, rs_vol, 0)
    else:
        print("  ! Rogers-Satchell column missing. Fallback to squared returns.")
        df['Var_Total'] = df['Ret']**2
        df['Var_Down'] = np.where(df['Ret'] < 0, df['Ret']**2, 0)
        df['Var_Up'] = np.where(df['Ret'] > 0, df['Ret']**2, 0)
    
    # Log-Transformation (Target & Features)
    eps = 1e-8 
    
    # 1. Total Volatility Setup
    df['Target_Total'] = np.log(df['Var_Total'].replace(0, eps)).shift(-1)
    df['Current_LogVol'] = np.log(df['Var_Total'].replace(0, eps))
    
    df['HAR_Total_d'] = np.log(df['Var_Total'].replace(0, eps))
    df['HAR_Total_w'] = np.log(df['Var_Total'].rolling(7).mean().replace(0, eps))
    df['HAR_Total_m'] = np.log(df['Var_Total'].rolling(30).mean().replace(0, eps))
    
    # 2. Downside Volatility Setup
    df['Target_Down'] = np.log(df['Var_Down'].replace(0, eps)).shift(-1)
    df['HAR_Down_d'] = np.log(df['Var_Down'].replace(0, eps))
    df['HAR_Down_w'] = np.log(df['Var_Down'].rolling(7).mean().replace(0, eps))
    df['HAR_Down_m'] = np.log(df['Var_Down'].rolling(30).mean().replace(0, eps))

    # 3. Upside Volatility Setup
    df['Target_Upside'] = np.log(df['Var_Up'].replace(0, eps)).shift(-1)
    df['HAR_Upside_d'] = np.log(df['Var_Up'].replace(0, eps))
    df['HAR_Upside_w'] = np.log(df['Var_Up'].rolling(7).mean().replace(0, eps))
    df['HAR_Upside_m'] = np.log(df['Var_Up'].rolling(30).mean().replace(0, eps))

    # Predictor Engineering
    models = {}
    # Note: 'FUND_BTC_Volume' (if present) will be captured by 'FUND_'
    pred_cols = [c for c in df.columns if any(x in c for x in ['GT_', 'WIKI_', 'FUND_', 'MKT_'])]
    
    print(f"  > Transforming {len(pred_cols)} predictors (Inc. Volume if avail)...")
    for col in pred_cols:
        try:
            # Differentiate stationary vs non-stationary series
            if (df[col] <= 0).any():
                df[f'X_{col}'] = df[col].diff()
            else:
                df[f'X_{col}'] = np.log(df[col]).diff()
            models[col] = [f'X_{col}']
        except Exception as e:
            print(f"    ! Skipping {col}: {e}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df, models

# ==========================================
# 2. FORECASTING ENGINE
# ==========================================
def run_forecast_routine(df, models, target_col, har_cols, label, window_size):
    print(f"\n--- Running Forecasts: {label} (W={window_size}) ---")
    
    oos_indices = range(window_size, len(df))
    
    preds = {m: [] for m in models.keys()}
    preds['HAR'] = []
    preds['Actual_LogVol'] = []
    preds['Prev_LogVol'] = [] 
    
    arr_target = df[target_col].values
    arr_har = df[har_cols].values
    arr_prev = df['Current_LogVol'].values 
    
    # Initialize progress bar if running main study window
    iterator = tqdm(oos_indices, unit="day", desc=f"W={window_size}") if window_size == CONFIG['window_size'] else oos_indices
    
    for t in iterator:
        train_start = t - window_size
        train_end = t
        
        y_train = arr_target[train_start:train_end]
        X_har_train = arr_har[train_start:train_end]
        X_har_train_c = np.column_stack([np.ones(len(X_har_train)), X_har_train])
        
        # HAR Benchmark Fit
        try:
            beta_h, _, _, _ = np.linalg.lstsq(X_har_train_c, y_train, rcond=None)
        except: 
            # Fallback for rare singular matrix
            beta_h = np.zeros(X_har_train_c.shape[1])
        
        X_har_test = np.append(1, arr_har[t])
        pred_har = np.dot(X_har_test, beta_h)
        
        preds['HAR'].append(pred_har)
        preds['Actual_LogVol'].append(arr_target[t])
        preds['Prev_LogVol'].append(arr_prev[t])
        
        # Augmented Model Fits
        for name, cols in models.items():
            raw_feat = df[cols].values
            f_train = raw_feat[train_start:train_end, 0]
            f_test_val = raw_feat[t, 0]
            
            # Winsorization to prevent outliers
            lower, upper = np.percentile(f_train, [1, 99])
            f_train_cl = np.clip(f_train, lower, upper)
            f_test_cl = np.clip(f_test_val, lower, upper)
            
            X_aug_train = np.column_stack([X_har_train_c, f_train_cl])
            X_aug_test = np.append(X_har_test, f_test_cl)
            
            try:
                beta_aug, _, _, _ = np.linalg.lstsq(X_aug_train, y_train, rcond=None)
                pred_aug = np.dot(X_aug_test, beta_aug)
                preds[name].append(pred_aug)
            except:
                preds[name].append(pred_har)

    return pd.DataFrame(preds, index=df.index[oos_indices])

# ==========================================
# 3. STATISTICAL EVALUATION
# ==========================================
def calculate_stats(res, label):
    actual = res['Actual_LogVol']
    har = res['HAR']
    prev = res['Prev_LogVol']
    
    mse_har = np.mean((actual - har)**2)
    actual_dir = (actual > prev).astype(int) 
    har_dir = (har > prev).astype(int)
    har_acc = np.mean(actual_dir == har_dir)
    
    stats_data = []
    
    for m in res.columns:
        if m in ['Actual_LogVol', 'Prev_LogVol']: continue
        
        # MSE / R2 Metrics
        mse = np.mean((actual - res[m])**2)
        r2 = 1 - (mse / mse_har)
        
        # Diebold-Mariano Test (Standard)
        dm_stat, p_val = 0, 1.0
        
        if m != 'HAR':
            e_bench = actual - har
            e_model = actual - res[m]
            
            # Loss Differential: d = Loss(Benchmark) - Loss(Model)
            d = (e_bench**2) - (e_model**2)
            
            if np.std(d) > 0:
                # Regress loss differential on a constant
                X = np.ones(len(d))
    
                # Use HAC (Newey-West) standard errors
                # maxlags=1 is standard for 1-step ahead, use int(T**(1/3)) for longer horizons
                model = sm.OLS(d, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    
                dm_stat = model.tvalues[0]
    
                # One-sided p-value (Upper tail: Model beats HAR)
                # Uses Student's t distribution (more robust than norm.cdf for finite samples)
                p_val = 1 - stats.t.cdf(dm_stat, df=model.df_resid)
                
        # Market Timing (Directional Accuracy)
        pred_dir = (res[m] > prev).astype(int)
        acc = np.mean(actual_dir == pred_dir)
        da_edge = (acc - har_acc) * 100
        
        stats_data.append({
            'Model': m, 
            'Type': label,
            'R2_OOS (%)': r2*100, 
            'MSE': mse, 
            'P_Val': p_val,
            'DA_Accuracy (%)': acc * 100,
            'DA_Edge_vs_HAR': da_edge
        })
        
    return pd.DataFrame(stats_data).sort_values('R2_OOS (%)', ascending=False)

# ==========================================
# 4. STRUCTURAL BREAK ANALYSIS (REVISED)
# ==========================================
def analyze_structural_break(res_tot, res_down, res_up):
    b_date = CONFIG['break_date']
    print(f"\n--- 4. Structural Break Analysis (Total, Down, Up) ---")
    
    combined_rankings = []
    
    # Process all three volatility types
    for vol_label, res_df in [('Total', res_tot), ('Downside', res_down), ('Upside', res_up)]:
        if not isinstance(res_df.index, pd.DatetimeIndex):
            res_df.index = pd.to_datetime(res_df.index)
            
        # Analyze three specific windows for each volatility type
        windows = [
            ('Full Period', res_df),
            ('Pre-Ban', res_df[res_df.index < b_date]),
            ('Post-Ban', res_df[res_df.index >= b_date])
        ]
        
        for period_label, chunk in windows:
            if len(chunk) < 10: continue
            
            # Generate full model rankings for this period/vol-type combination
            stats_df = calculate_stats(chunk, f"{vol_label}_{period_label}")
            stats_df['Volatility_Type'] = vol_label
            stats_df['Period'] = period_label
            combined_rankings.append(stats_df)
            
    full_breakdown = pd.concat(combined_rankings, ignore_index=True)
    
    # --- PLOT LOGIC (Enriched Labels & Including Volume) ---
    res_df_plot = res_tot
    pre_df = res_df_plot[res_df_plot.index < b_date]
    post_df = res_df_plot[res_df_plot.index >= b_date]
    
    plot_results = []
    cols = res_df_plot.columns
    
    # Identify key variables
    inf_cols = [c for c in cols if 'Inflation' in c]
    best_inf = inf_cols[0] if inf_cols else None
    
    btc_cols = [c for c in cols if 'Bitcoin' in c and 'GT_' in c]
    best_btc = btc_cols[0] if btc_cols else None
    
    vol_col = 'FUND_BTC_Volume' if 'FUND_BTC_Volume' in cols else None
    gold_model = 'FUND_Gold' if 'FUND_Gold' in cols else None
    
    # Select models to compare
    models_to_test = [m for m in [best_inf, best_btc, gold_model, vol_col] if m]
    
    # Enhanced Labeling Function
    def clean_sb_label(s):
        if 'GT_' in s: return f"Retail Hype\n(Google: {s.replace('GT_', '')})"
        if 'WIKI_' in s: return f"Macro Fear\n(Wiki: {s.replace('WIKI_', '')})"
        if 'FUND_BTC_Volume' in s: return "Trading Activity\n(Volume)"
        if 'FUND_' in s: return f"Fund: {s.replace('FUND_', '')}"
        return s.replace('MKT_', '')

    for period, chunk in [('Pre-Ban (Retail Era)', pre_df), ('Post-Ban (Macro Era)', post_df)]:
        if len(chunk) < 10: continue
        actual = chunk['Actual_LogVol']
        har_mse = np.mean((actual - chunk['HAR'])**2)
        for m in models_to_test:
            mse = np.mean((actual - chunk[m])**2)
            r2 = 1 - (mse / har_mse)
            plot_results.append({'Period': period, 'Model': m, 'R2_OOS (%)': r2 * 100})
            
    res_table = pd.DataFrame(plot_results)
    if not res_table.empty:
        plt.figure(figsize=(11, 6))
        res_table['Label'] = res_table['Model'].apply(clean_sb_label)
        
        sns.barplot(data=res_table, x='Label', y='R2_OOS (%)', hue='Period', 
                    palette={'Pre-Ban (Retail Era)': '#9467bd', 'Post-Ban (Macro Era)': '#2ca02c'})
        
        plt.axhline(0, color='black', lw=1)
        plt.title(f"Structural Break: Impact of China Mining Ban ({b_date})", fontweight='bold')
        plt.ylabel("Forecasting Gain vs HAR (R² %)")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['output_dir'], 'Fig7_Structural_Break.pdf'))
        plt.close()
        
    return full_breakdown

# ==========================================
# 5. ROBUSTNESS CHECK
# ==========================================
def check_robustness(df, models, target_col, har_cols):
    print("\n--- 5. Robustness Check (Sensitivity to Window Size) ---")
    
    windows = [365, 730, 1095]
    summary = []
    
    inf_cols = [c for c in models.keys() if 'Inflation' in c and 'WIKI' in c]
    tgt_model = inf_cols[0] if inf_cols else list(models.keys())[0]
    
    print(f"  > Tracking Model: {tgt_model}")
    
    for w in windows:
        print(f"    > Testing Window W={w}...")
        res = run_forecast_routine(df, models, target_col, har_cols, 'Robustness', w)
        stats_df = calculate_stats(res, 'Robustness')
        
        stats_df['Rank'] = range(1, len(stats_df) + 1)
        row = stats_df[stats_df['Model'] == tgt_model].iloc[0]
        
        summary.append({
            'Window (Days)': w,
            'Model': tgt_model,
            'R2_OOS (%)': row['R2_OOS (%)'],
            'Rank': row['Rank'],
            'P_Val': row['P_Val']
        })
        
    rob_df = pd.DataFrame(summary)
    return rob_df

# ==========================================
# 6. VISUALIZATION SUITE
# ==========================================
def generate_visuals(res_total, stats_total, stats_down, stats_up):
    print("\n--- Generating Visuals ---")
    
    winner = stats_total.iloc[0]['Model']
    infl_list = [m for m in stats_total['Model'] if 'Inflation' in m]
    best_infl = infl_list[0] if infl_list else winner
    
    # --- PLOT 1: Cumulative Squared Prediction Error (CSPE) ---
    actual = res_total['Actual_LogVol']
    har_err = (actual - res_total['HAR'])**2
    
    plt.figure(figsize=(12, 7))
    plt.axhline(0, color='black', lw=1)
    
    # Plot Winner
    d_win = (har_err - (actual - res_total[winner])**2).cumsum()
    plt.plot(d_win, label=f"Top Model: {winner}", color='#2ca02c', lw=2)
    
    # Plot Retail Benchmark (Google Bitcoin)
    btc_model = 'GT_Bitcoin'
    if btc_model in res_total.columns and winner != btc_model:
        d_btc = (har_err - (actual - res_total[btc_model])**2).cumsum()
        plt.plot(d_btc, label=f"Retail: {btc_model}", color='#ff7f0e', lw=2, linestyle='--')
        
    # Plot Volume if not winner (Added Comparison)
    vol_model = 'FUND_BTC_Volume'
    if vol_model in res_total.columns and winner != vol_model:
        d_vol = (har_err - (actual - res_total[vol_model])**2).cumsum()
        plt.plot(d_vol, label=f"Volume", color='#9467bd', lw=2, linestyle=':')
        
    plt.title("Cumulative Forecast Gain (Total Volatility) vs HAR", fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CONFIG['output_dir'], 'Fig1_Horse_Race.pdf'))
    plt.close()

    # --- PLOT 2: Asymmetry Test (Upside vs Downside) ---
    s_up = stats_up.set_index('Model')[['R2_OOS (%)']].rename(columns={'R2_OOS (%)': 'Upside Vol'})
    s_down = stats_down.set_index('Model')[['R2_OOS (%)']].rename(columns={'R2_OOS (%)': 'Downside Vol'})
    
    # Added 'FUND_BTC_Volume' to comparison
    comp_list = ['WIKI_Inflation', 'GT_Bitcoin', 'MKT_VIX', 'FUND_BTC_Volume']
    key_models = [m for m in comp_list if m in s_up.index]
    
    comp = s_up.join(s_down, how='inner')
    comp = comp.loc[comp.index.intersection(key_models)]
    
    if not comp.empty:
        comp.plot(kind='bar', figsize=(10, 6), color=['#2ca02c', '#d62728'], width=0.7)
        plt.axhline(0, color='black', lw=1)
        plt.title("Asymmetry: Upside vs Downside Volatility Prediction", fontweight='bold')
        plt.ylabel("R2 OOS (%)")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['output_dir'], 'Fig5_Asymmetry_Test.pdf'))
        plt.close()

    # --- PLOT 3: Directional Accuracy ---
    top_5 = stats_total.head(5).copy()
    har_row = stats_total[stats_total['Model'] == 'HAR'].iloc[0]
    
    plt.figure(figsize=(10, 6))
    
    def format_distinct_label(s):
        if 'WIKI_' in s: return f"Inflation\n(Wiki)" if 'Inflation' in s else f"{s.replace('WIKI_', '')}\n(Wiki)"
        if 'GT_' in s: return f"Inflation\n(GT)" if 'Inflation' in s else f"{s.replace('GT_', '')}\n(GT)"
        if 'FUND_BTC_Volume' in s: return "Trading Activity\n(Volume)"
        if 'FUND_' in s: return f"{s.replace('FUND_', '')}\n(Fund)"
        return s.replace('MKT_', '')

    top_5['Label'] = top_5['Model'].apply(format_distinct_label)
    
    bars = plt.bar(top_5['Label'], top_5['DA_Accuracy (%)'], color='#2ca02c', alpha=0.8)
    plt.axhline(har_row['DA_Accuracy (%)'], color='black', ls='--', lw=2, label=f"HAR Baseline ({har_row['DA_Accuracy (%)']:.1f}%)")
    
    plt.ylim(50, max(top_5['DA_Accuracy (%)'].max(), har_row['DA_Accuracy (%)']) + 2)
    plt.ylabel("Directional Accuracy (%)")
    plt.title("Market Timing: Accuracy of predicting Volatility Direction (Up/Down)", fontweight='bold')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}%', ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'Fig6_Directional_Accuracy.pdf'))
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    df, models = prep_data()
    
    # 1. Forecast Routines
    res_tot = run_forecast_routine(df, models, 'Target_Total', 
                                   ['HAR_Total_d', 'HAR_Total_w', 'HAR_Total_m'], 
                                   'Total', CONFIG['window_size'])
    stats_tot = calculate_stats(res_tot, 'Total')
    
    res_down = run_forecast_routine(df, models, 'Target_Down', 
                                    ['HAR_Down_d', 'HAR_Down_w', 'HAR_Down_m'], 
                                    'Downside', CONFIG['window_size'])
    stats_down = calculate_stats(res_down, 'Downside')
    
    res_up = run_forecast_routine(df, models, 'Target_Upside', 
                                  ['HAR_Upside_d', 'HAR_Upside_w', 'HAR_Upside_m'], 
                                  'Upside', CONFIG['window_size'])
    stats_up = calculate_stats(res_up, 'Upside')
    
    # 2. Structural Break (Revised to analyze all combinations)
    break_results = analyze_structural_break(res_tot, res_down, res_up)
    
    # 3. Robustness
    robust_results = check_robustness(df, models, 'Target_Total', 
                                      ['HAR_Total_d', 'HAR_Total_w', 'HAR_Total_m'])
    
    # 4. Save Final Results
    print("\n--- Saving Results ---")
    stats_tot.to_csv(os.path.join(CONFIG['output_dir'], 'rankings_total.csv'), index=False)
    stats_down.to_csv(os.path.join(CONFIG['output_dir'], 'rankings_downside.csv'), index=False)
    stats_up.to_csv(os.path.join(CONFIG['output_dir'], 'rankings_upside.csv'), index=False)
    
    res_tot.to_csv(os.path.join(CONFIG['output_dir'], 'predictions_total.csv'))
    res_down.to_csv(os.path.join(CONFIG['output_dir'], 'predictions_downside.csv'))
    res_up.to_csv(os.path.join(CONFIG['output_dir'], 'predictions_upside.csv'))
    
    break_results.to_csv(os.path.join(CONFIG['output_dir'], 'structural_break_full_breakdown.csv'), index=False)
    robust_results.to_csv(os.path.join(CONFIG['output_dir'], 'robustness_check.csv'), index=False)
    
    # 5. Visuals
    generate_visuals(res_tot, stats_tot, stats_down, stats_up)
    
    print("\n=== STUDY COMPLETE ===")
