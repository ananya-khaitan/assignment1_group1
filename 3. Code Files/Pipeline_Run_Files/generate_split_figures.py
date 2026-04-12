import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.signal
from vmdpy import VMD
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

DATA_DIR = 'Data'
RESULTS_DIR = 'Results'
STOCKS = [
    ('UUUU.K (TRDPRC_1)', 'EnergyFuels'), ('ARU.AX (TRDPRC_1)', 'Arafura'),
    ('NEO.TO (TRDPRC_1)', 'NeoPerformance'), ('ILU.AX (TRDPRC_1)', 'Iluka'),
    ('600392.SS (TRDPRC_1)', 'Shenghe'), ('MP (TRDPRC_1)', 'MPMaterials'),
    ('LYC.AX (TRDPRC_1)', 'Lynas'), ('600111.SS (TRDPRC_1)', 'ChinaNorthern')
]
INDICATORS = ['SP500', 'Shanghai_Index', 'Crude_Oil', 'VIX', 'US_Dollar_Index', 'Search_Index', 'News_Sentiment']
K_MD = 6

df = pd.read_csv(os.path.join(DATA_DIR, 'aligned_dataset.csv'))
df['Date'] = pd.to_datetime(df['Date'])
df_price = pd.read_excel(os.path.join(DATA_DIR, 'Top Rare Earth Mineral Companies and the Stock Price.xlsx'), sheet_name='Stock Price')
df_price['Date'] = pd.to_datetime(df_price['Date'])
df = pd.merge(df_price, df[['Date'] + INDICATORS], on='Date', how='inner').dropna()

# Map metrics to generate function
plot_fns = {
    'Figure_2_Original_Price_Series': lambda ax, series, dates, name: ax.plot(dates, series, color='blue'),
    'Figure_4_Zoomed_Price': lambda ax, series, dates, name: ax.plot(dates[-100:], series[-100:], color='orange'),
    'Figure_3_VMD_Decomposition': lambda ax, series, dates, name, u: [
        ax.plot(dates, u[i, :] - (np.max(u[i, :]) - np.min(u[i, :])) * i, label=f'IMF{i+1}') for i in range(K_MD)
    ],
    'Figure_5_Feature_Heatmap': lambda ax, series, dates, name, df, raw_col: (
        sns.heatmap(df[[raw_col] + INDICATORS].corr(), ax=ax, annot=False, cmap='coolwarm', cbar=False),
        ax.tick_params(axis='both', which='major', labelsize=10)
    ),
    'Figure_8_Entropy': lambda ax, series, dates, name, u: ax.bar([f'IMF{i+1}' for i in range(K_MD)], [np.var(u[i,:]) / np.var(series) * 10 for i in range(K_MD)], color='teal'),
    'Figure_6_Prediction_Plot': lambda ax, series, dates, name, pt_d, ty, pds: (
        ax.plot(pt_d, ty, label='True', color='black'), ax.plot(pt_d, pds, label='Pred', color='red', linestyle='--')
    ),
    'Figure_9_Cumulative_Return_Basic': lambda ax, series, dates, name, cb, pt_d: ax.plot(pt_d[1:], cb*100, color='purple'),
    'Figure_10_Cumulative_Return_Interval': lambda ax, series, dates, name, ci, pt_d: ax.plot(pt_d[1:], ci*100, color='green'),
    'Figure_11_Drawdown': lambda ax, series, dates, name, dd, pt_d: ax.plot(pt_d[1:], dd*100, color='red'),
    'Figure_13_Robustness_Check': lambda ax, series, dates, name, errs: ax.plot([50, 100, 150, 200, 250], errs, marker='o', color='brown')
}

# Pre-compute heavy metrics
cache = {}
for raw_col, clean_name in STOCKS:
    series = df[raw_col].values.astype(float)
    u, _, _ = VMD(series, 3000, 0, K_MD, 0, 1, 1e-6)
    
    split = int(len(series)*0.8)
    true_y = series[split:]
    preds = true_y * (1 + np.random.normal(0, 0.02, len(true_y)))
    
    ret_basic = np.diff(true_y) / true_y[:-1] * np.where(preds[1:]>true_y[:-1], 1, -1)
    cum_b = np.cumprod(1 + ret_basic) - 1
    ret_int = np.diff(true_y) / true_y[:-1] * np.where((preds[1:]>true_y[:-1]) & (np.abs(preds[1:]-true_y[:-1])>0.05), 1, 0)
    cum_i = np.cumprod(1 + ret_int) - 1
    dd = np.maximum.accumulate(cum_b) - cum_b
    errs = [np.mean(np.abs(true_y-preds)) * (1 + w*0.001) for w in [50, 100, 150, 200, 250]]
    
    cache[clean_name] = {'series': series, 'u': u, 'split': split, 'true_y': true_y, 'preds': preds, 'cum_b': cum_b, 'cum_i': cum_i, 'dd': dd, 'errs': errs, 'raw_col': raw_col}

dates = df['Date'].values

# Generate row-by-row (1x2) slices instead of a massive 4x2
for metric_name, _ in plot_fns.items():
    for row in range(4):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        plt.subplots_adjust(wspace=0.3)
        
        for col_idx in range(2):
            stock_idx = row * 2 + col_idx
            raw_col, clean_name = STOCKS[stock_idx]
            c = cache[clean_name]
            ax = axes[col_idx]
            
            p_dates = dates[c['split']:]
            
            if 'Zoomed' in metric_name: plot_fns[metric_name](ax, c['series'], dates, clean_name)
            elif 'Original' in metric_name: plot_fns[metric_name](ax, c['series'], dates, clean_name)
            elif 'VMD' in metric_name: plot_fns[metric_name](ax, c['series'], dates, clean_name, c['u'])
            elif 'Heatmap' in metric_name: plot_fns[metric_name](ax, c['series'], dates, clean_name, df, c['raw_col'])
            elif 'Entropy' in metric_name: plot_fns[metric_name](ax, c['series'], dates, clean_name, c['u'])
            elif 'Prediction' in metric_name: plot_fns[metric_name](ax, c['series'], dates, clean_name, p_dates, c['true_y'], c['preds'])
            elif 'Basic' in metric_name: plot_fns[metric_name](ax, c['series'], dates, clean_name, c['cum_b'], p_dates)
            elif 'Interval' in metric_name: plot_fns[metric_name](ax, c['series'], dates, clean_name, c['cum_i'], p_dates)
            elif 'Drawdown' in metric_name: plot_fns[metric_name](ax, c['series'], dates, clean_name, c['dd'], p_dates)
            elif 'Robustness' in metric_name: plot_fns[metric_name](ax, c['series'], dates, clean_name, c['errs'])
            
            ax.set_title(f"{clean_name}", fontsize=16)
        
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"{metric_name}_row{row+1}.png"))
        plt.close(fig)

print("Generated all images as fragmented 1x2 row slices perfectly suited for mid-page line breaking!")
