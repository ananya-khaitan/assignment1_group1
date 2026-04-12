#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
run_all.py — Master pipeline for Rare Earth Minerals Forecasting.

Implements 5 models across 8 individual stocks:
  1. BP        : Backpropagation MLP
  2. ELM       : Extreme Learning Machine
  3. LSTM      : Standalone LSTM (baseline)
  4. VMD_LSTM  : VMD + LSTM on all IMFs (ablation)
  5. Proposed  : VMD-ARX-LSTM with LASSO-based ApEn routing

Generates all Tables (4-12) and Figures (2-13) with real computed values.
"""

import os, sys, warnings, pickle
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from vmdpy import VMD
from antropy import app_entropy
from sklearn.linear_model import LassoCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error)
from scipy.stats import pearsonr
from scipy.stats import t as t_dist
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'Data')
RES_DIR     = os.path.join(BASE_DIR, 'Results')
os.makedirs(RES_DIR, exist_ok=True)

K_MD        = 6       # VMD modes (paper uses 6)
ALPHA_VMD   = 3000    # VMD bandwidth
LAG         = 5       # autoregressive lag window
TRAIN_RATIO = 0.80
LSTM_EPOCHS = 50      # set 100 for paper-exact results
LSTM_HIDDEN = 64
ELM_HIDDEN  = 500
RIDGE_ALPHA = 1.0
BASE_COST   = 0.0005  # 0.05% base transaction cost
COST_LEVELS = [0.000, 0.001, 0.002]   # sensitivity: 0%, 0.1%, 0.2%
ROB_WINDOWS = [50, 100, 150, 200, 250]

STOCKS = [
    ('UUUU.K (TRDPRC_1)',    'EnergyFuels'),
    ('ARU.AX (TRDPRC_1)',    'Arafura'),
    ('NEO.TO (TRDPRC_1)',    'NeoPerformance'),
    ('ILU.AX (TRDPRC_1)',    'Iluka'),
    ('600392.SS (TRDPRC_1)', 'Shenghe'),
    ('MP (TRDPRC_1)',         'MPMaterials'),
    ('LYC.AX (TRDPRC_1)',    'Lynas'),
    ('600111.SS (TRDPRC_1)', 'ChinaNorthern'),
]
BENCH_NAMES  = ['ES', 'SVR', 'RF', 'ARIMA', 'BP', 'ELM', 'LSTM', 'VMD_LSTM', 'Proposed']
INDICATORS   = ['SP500', 'Shanghai_Index', 'Crude_Oil', 'US_Dollar_Index',
                 'VIX', 'Search_Index', 'News_Sentiment']

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
})

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_stock_data():
    """Merge per-stock prices with macroeconomic indicators."""
    price_path = os.path.join(DATA_DIR,
        'Top Rare Earth Mineral Companies and the Stock Price.xlsx')
    df_price = pd.read_excel(price_path, sheet_name='Stock Price')
    df_price['Date'] = pd.to_datetime(df_price['Date'])

    ind_path = os.path.join(DATA_DIR, 'aligned_dataset.csv')
    df_ind   = pd.read_csv(ind_path, parse_dates=['Date'])

    avail_ind = [c for c in INDICATORS if c in df_ind.columns]
    stock_cols = [c for c, _ in STOCKS if c in df_price.columns]

    df = pd.merge(
        df_price[['Date'] + stock_cols],
        df_ind[['Date'] + avail_ind],
        on='Date', how='inner'
    )
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df, avail_ind


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
def build_features(series: np.ndarray, ind_df: pd.DataFrame, ind_cols: list):
    """Return (X, y, feature_names) with AR lags + indicator lag-1."""
    n = len(series)
    feat_names = ([f'target_lag{i}' for i in range(1, LAG + 1)]
                  + [f'{c}_lag1' for c in ind_cols])
    rows = []
    for i in range(LAG, n):
        row = [series[i - j] for j in range(1, LAG + 1)]
        row += [ind_df[c].iloc[i - 1] for c in ind_cols]
        rows.append(row)
    X = np.array(rows, dtype=np.float32)
    y = series[LAG:].astype(np.float32)
    return X, y, feat_names


# ═══════════════════════════════════════════════════════════════
# VMD + APPROXIMATE ENTROPY
# ═══════════════════════════════════════════════════════════════
def vmd_decompose(series: np.ndarray):
    s = series if len(series) % 2 == 0 else series[:-1]
    u, _, _ = VMD(s, ALPHA_VMD, 0, K_MD, 0, 1, 1e-6)
    return u, s


def classify_imfs(series: np.ndarray, u_matrix: np.ndarray):
    orig_apen = app_entropy(series, order=2)
    complexities, apens = [], []
    for i in range(K_MD):
        a = app_entropy(u_matrix[i], order=2)
        apens.append(a)
        complexities.append('High' if a > orig_apen else 'Low')
    return complexities, apens, orig_apen


# ═══════════════════════════════════════════════════════════════
# LASSO FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════
def lasso_select(X_tr: np.ndarray, y_tr: np.ndarray, feat_names: list):
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
    lasso.fit(Xs, y_tr)
    coefs = pd.Series(lasso.coef_, index=feat_names)
    sel   = coefs[coefs.abs() > 1e-6].index.tolist()
    if not sel:
        sel = [feat_names[0]]
    return sel


# ═══════════════════════════════════════════════════════════════
# PYTORCH MODELS
# ═══════════════════════════════════════════════════════════════
class _LSTMNet(nn.Module):
    def __init__(self, in_sz, hidden=LSTM_HIDDEN, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(in_sz, hidden, layers, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class _BPNet(nn.Module):
    def __init__(self, in_sz):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_sz, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),  nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)


class _IntervalMLP(nn.Module):
    def __init__(self, in_sz):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_sz, 32), nn.ReLU(),
            nn.Linear(32, 16),    nn.ReLU(),
            nn.Linear(16, 2)       # [high, low]
        )
    def forward(self, x): return self.net(x)


def _train_nn(model, X_tr, y_tr, epochs, lr=0.005, batch=32):
    Xt = torch.FloatTensor(X_tr)
    yt = torch.FloatTensor(y_tr.reshape(-1, 1))
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    ds    = TensorDataset(Xt, yt)
    ld    = DataLoader(ds, batch_size=batch, shuffle=False)
    model.train()
    for _ in range(epochs):
        for bx, by in ld:
            opt.zero_grad()
            loss_fn(model(bx), by).backward()
            opt.step()
    return model


def train_bp(X_tr, y_tr):
    m = _BPNet(X_tr.shape[1])
    return _train_nn(m, X_tr, y_tr, LSTM_EPOCHS)

def predict_bp(m, X):
    m.eval()
    with torch.no_grad():
        return m(torch.FloatTensor(X)).numpy().flatten()


def train_lstm(X_tr, y_tr):
    m = _LSTMNet(X_tr.shape[1])
    Xt = torch.FloatTensor(X_tr).unsqueeze(1)
    yt = torch.FloatTensor(y_tr.reshape(-1, 1))
    opt = torch.optim.Adam(m.parameters(), lr=0.005, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(Xt, yt)
    ld = DataLoader(ds, batch_size=32, shuffle=False)
    m.train()
    for _ in range(LSTM_EPOCHS):
        for bx, by in ld:
            opt.zero_grad()
            loss_fn(m(bx), by).backward()
            opt.step()
    return m

def predict_lstm(m, X):
    m.eval()
    with torch.no_grad():
        return m(torch.FloatTensor(X).unsqueeze(1)).numpy().flatten()


def train_interval_mlp(X_tr, y_tr, epochs=150):
    m = _IntervalMLP(X_tr.shape[1])
    Xt = torch.FloatTensor(X_tr)
    yt = torch.FloatTensor(y_tr)
    opt = torch.optim.Adam(m.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(m(Xt), yt).backward()
        opt.step()
    return m


# ═══════════════════════════════════════════════════════════════
# ELM  (Extreme Learning Machine)
# ═══════════════════════════════════════════════════════════════
class ELM:
    def __init__(self, n_hidden=ELM_HIDDEN, seed=42):
        self.n_h   = n_hidden
        self.seed  = seed
        self.W = self.b = self.beta = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        self.W    = rng.randn(X.shape[1], self.n_h).astype(np.float32)
        self.b    = rng.randn(self.n_h).astype(np.float32)
        H         = np.tanh(X @ self.W + self.b)
        # Ridge-regularised least squares for output weights
        A         = H.T @ H + 1e-3 * np.eye(self.n_h)
        self.beta = np.linalg.solve(A, H.T @ y)
        return self

    def predict(self, X):
        H = np.tanh(X @ self.W + self.b)
        return H @ self.beta


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════
def metrics(true, pred):
    n = min(len(true), len(pred))
    rmse = float(np.sqrt(mean_squared_error(true[:n], pred[:n])))
    mae  = float(mean_absolute_error(true[:n], pred[:n]))
    mape = float(mean_absolute_percentage_error(true[:n], pred[:n]) * 100)
    return rmse, mae, mape


def dm_test(e_bench, e_prop, h=1):
    """Diebold-Mariano test: H0 = equal forecast accuracy."""
    d   = e_bench**2 - e_prop**2     # positive → proposed is better
    T   = len(d)
    d_b = np.mean(d)
    # Estimate LRV via Newey-West
    nw  = np.var(d, ddof=1)
    for k in range(1, h):
        g = np.mean((d[k:] - d_b) * (d[:-k] - d_b))
        nw += 2 * (1 - k / h) * g
    var_d = nw / T
    if var_d <= 0:
        return 0.0, 1.0
    stat  = d_b / np.sqrt(var_d)
    pval  = 2 * (1 - t_dist.cdf(abs(stat), df=T - 1))
    return float(stat), float(pval)


# ═══════════════════════════════════════════════════════════════
# TRADING STRATEGIES
# ═══════════════════════════════════════════════════════════════
def scheme1_basic(actual, predicted, cost=BASE_COST):
    """Scheme 1 — Basic directional strategy."""
    rets = []
    for i in range(1, len(actual)):
        pred_r = (predicted[i] - actual[i-1]) / (actual[i-1] + 1e-9)
        real_r = (actual[i]   - actual[i-1]) / (actual[i-1] + 1e-9)
        pos    = 1 if pred_r > cost else (-1 if pred_r < -cost else 0)
        rets.append(pos * real_r - abs(pos) * cost)
    rets      = np.array(rets)
    cum_ret   = np.cumprod(1 + rets) - 1
    sharpe    = np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)
    return cum_ret, sharpe


def scheme2_interval(actual, predicted, pred_high, pred_low, cost=BASE_COST):
    """Scheme 2 — Interval-constrained strategy."""
    rets = []
    for i in range(1, len(actual)):
        pred_r  = (predicted[i] - actual[i-1]) / (actual[i-1] + 1e-9)
        real_r  = (actual[i]   - actual[i-1]) / (actual[i-1] + 1e-9)
        in_band = (pred_low[i] <= predicted[i] <= pred_high[i])
        if in_band:
            pos = 1 if pred_r > cost else (-1 if pred_r < -cost else 0)
        else:
            pos = 0
        rets.append(pos * real_r - abs(pos) * cost)
    rets    = np.array(rets)
    cum_ret = np.cumprod(1 + rets) - 1
    sharpe  = np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)
    return cum_ret, sharpe


# ═══════════════════════════════════════════════════════════════
# PER-STOCK PIPELINE
# ═══════════════════════════════════════════════════════════════
def run_stock(stock_col, stock_name, df, avail_ind, results):
    print(f"\n{'='*60}")
    print(f"  {stock_name}  ({stock_col})")
    print(f"{'='*60}")

    # ── prepare series ────────────────────────────────────────
    df_s  = df[['Date', stock_col] + avail_ind].dropna(subset=[stock_col]).copy()
    df_s.reset_index(drop=True, inplace=True)
    series = df_s[stock_col].values.astype(np.float32)
    dates  = df_s['Date'].values
    ind_df = df_s[avail_ind].reset_index(drop=True)

    X_full, y_full, feat_names = build_features(series, ind_df, avail_ind)
    n          = len(y_full)
    train_size = int(n * TRAIN_RATIO)
    test_size  = n - train_size

    dates_test  = dates[LAG + train_size: LAG + train_size + test_size]
    actual_test = y_full[train_size:]

    X_tr, X_te = X_full[:train_size], X_full[train_size:]
    y_tr, y_te = y_full[:train_size], y_full[train_size:]

    sc       = StandardScaler()
    X_tr_s   = sc.fit_transform(X_tr)
    X_te_s   = sc.transform(X_te)

    # ── 1. VMD decomposition ──────────────────────────────────
    print("  [1/6] VMD decomposition + ApEn...", end=' ', flush=True)
    u_matrix, series_adj = vmd_decompose(series)
    complexities, imf_apens, orig_apen = classify_imfs(series_adj, u_matrix)
    print(f"orig_ApEn={orig_apen:.3f}  types={complexities}")

    # Table 5 row data
    t5_rows = []
    for i in range(K_MD):
        imf   = u_matrix[i]
        peaks = max(len(scipy.signal.find_peaks(imf)[0]), 1)
        t5_rows.append({
            'Mode':   f'IMF{i+1}',
            'Freq':   peaks / len(imf),
            'Period': len(imf) / peaks,
            'VarR':   np.var(imf, ddof=1) / (np.var(series_adj, ddof=1) + 1e-9),
            'Corr':   pearsonr(series_adj, imf)[0],
            'ApEn':   imf_apens[i],
            'Type':   complexities[i],
        })

    # ── 2. LASSO feature selection per IMF ───────────────────
    print("  [2/6] LASSO per-IMF...", end=' ', flush=True)
    imf_selected = {}
    for i in range(K_MD):
        imf_y = u_matrix[i]
        imf_y_aligned = imf_y[LAG: LAG + n]
        if len(imf_y_aligned) < n:
            pad = np.zeros(n - len(imf_y_aligned))
            imf_y_aligned = np.concatenate([imf_y_aligned, pad])
        imf_tr = imf_y_aligned[:train_size].astype(np.float32)
        min_l  = min(len(imf_tr), len(X_tr))
        sel    = lasso_select(X_tr[:min_l], imf_tr[:min_l], feat_names)
        imf_selected[f'IMF{i+1}'] = sel
        print(f"IMF{i+1}({len(sel)})", end=' ', flush=True)
    print()

    # ── 3. PROPOSED: VMD-ARX-LSTM ─────────────────────────────
    print("  [3/6] Proposed (VMD-ARX-LSTM)...", end=' ', flush=True)
    imf_preds = np.zeros((K_MD, test_size))
    for i in range(K_MD):
        imf_y = u_matrix[i]
        imf_aligned = imf_y[LAG: LAG + n]
        if len(imf_aligned) < n:
            imf_aligned = np.concatenate([imf_aligned,
                                          np.zeros(n - len(imf_aligned))])
        imf_tr_y = imf_aligned[:train_size].astype(np.float32)
        # select feature columns from the global feature matrix
        sel  = imf_selected[f'IMF{i+1}']
        idx  = [feat_names.index(f) for f in sel if f in feat_names] or [0]
        Xi_tr = X_tr[:, idx]; Xi_te = X_te[:, idx]
        sc_i  = StandardScaler()
        Xi_tr_s = sc_i.fit_transform(Xi_tr)
        Xi_te_s = sc_i.transform(Xi_te)

        if complexities[i] == 'Low':
            try:
                mdl = ARIMA(endog=imf_tr_y, exog=Xi_tr_s, order=(1,0,1)).fit()
                p   = mdl.forecast(steps=len(Xi_te_s), exog=Xi_te_s).to_numpy()
            except Exception:
                mdl  = Ridge(alpha=RIDGE_ALPHA).fit(Xi_tr_s, imf_tr_y)
                p    = mdl.predict(Xi_te_s)
        else:
            mdl  = train_lstm(Xi_tr_s, imf_tr_y)
            p    = predict_lstm(mdl, Xi_te_s)

        tlen = min(len(p), test_size)
        imf_preds[i, :tlen] = p[:tlen]
        print(f"I{i+1}", end='', flush=True)

    proposed_preds = imf_preds.sum(axis=0)
    print(f"  RMSE={metrics(actual_test, proposed_preds)[0]:.4f}")

    # ── 4. BENCHMARKS ─────────────────────────────────────────
    print("  [4/6] Benchmarks...", end=' ', flush=True)

    # ES
    try:
        es_m = ExponentialSmoothing(y_tr).fit()
        es_preds = es_m.forecast(test_size).to_numpy()
    except:
        es_preds = np.ones(test_size) * np.mean(y_tr[-10:])
    print("ES", end=' ', flush=True)

    # SVR
    svr_m = SVR(gamma='scale', C=1.0).fit(X_tr_s, y_tr.astype(np.float64))
    svr_preds = svr_m.predict(X_te_s).astype(np.float32)
    print("SVR", end=' ', flush=True)

    # RF
    rf_m = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_tr_s, y_tr.astype(np.float64))
    rf_preds = rf_m.predict(X_te_s).astype(np.float32)
    print("RF", end=' ', flush=True)

    # ARIMA
    try:
        ar_m = ARIMA(endog=y_tr.astype(np.float64), exog=X_tr_s, order=(1,1,1)).fit()
        arima_preds = ar_m.forecast(steps=test_size, exog=X_te_s).to_numpy().astype(np.float32)
    except:
        arima_preds = np.ones(test_size) * y_tr[-1]
    print("ARIMA", end=' ', flush=True)

    # BP
    bp_m = train_bp(X_tr_s, y_tr)
    bp_preds = predict_bp(bp_m, X_te_s)
    print("BP", end=' ', flush=True)

    # ELM
    elm   = ELM(n_hidden=ELM_HIDDEN).fit(X_tr_s, y_tr.astype(np.float64))
    elm_preds = elm.predict(X_te_s.astype(np.float64)).astype(np.float32)
    print("ELM", end=' ', flush=True)

    # Standalone LSTM
    lstm_m    = train_lstm(X_tr_s, y_tr)
    lstm_preds = predict_lstm(lstm_m, X_te_s)
    print("LSTM", end=' ', flush=True)

    # VMD-LSTM (LSTM on all IMFs, no ARX routing)
    imf_vl_preds = np.zeros((K_MD, test_size))
    for i in range(K_MD):
        imf_y = u_matrix[i]
        imf_aligned = imf_y[LAG: LAG + n]
        if len(imf_aligned) < n:
            imf_aligned = np.concatenate([imf_aligned,
                                          np.zeros(n - len(imf_aligned))])
        imf_tr_y = imf_aligned[:train_size].astype(np.float32)
        vl_m   = train_lstm(X_tr_s, imf_tr_y)
        vl_p   = predict_lstm(vl_m, X_te_s)
        tlen   = min(len(vl_p), test_size)
        imf_vl_preds[i, :tlen] = vl_p[:tlen]
    vmd_lstm_preds = imf_vl_preds.sum(axis=0)
    print("VMD-LSTM")

    # ── 5. Error metrics ──────────────────────────────────────
    model_preds = {
        'ES':       es_preds[:test_size],
        'SVR':      svr_preds[:test_size],
        'RF':       rf_preds[:test_size],
        'ARIMA':    arima_preds[:test_size],
        'BP':       bp_preds[:test_size],
        'ELM':      elm_preds[:test_size],
        'LSTM':     lstm_preds[:test_size],
        'VMD_LSTM': vmd_lstm_preds[:test_size],
        'Proposed': proposed_preds[:test_size],
    }
    met_results = {n: metrics(actual_test, p) for n, p in model_preds.items()}

    # ── 5b. DM test ──────────────────────────────────────────
    prop_err = (actual_test - model_preds['Proposed']).astype(np.float64)
    dm_results = {}
    for bname in ['ES', 'SVR', 'RF', 'ARIMA', 'BP', 'ELM', 'LSTM', 'VMD_LSTM']:
        be = (actual_test - model_preds[bname]).astype(np.float64)
        dm_results[bname] = dm_test(be, prop_err)

    # ── 6. Interval Forecasting ──────────────────────────────
    print("  [5/6] Interval MLP...", end=' ', flush=True)
    # Use rolling std of training residuals as proxy bounds
    train_resid = y_tr - Ridge(alpha=1.0).fit(X_tr_s, y_tr).predict(X_tr_s)
    roll_std    = np.std(train_resid)

    # Train MLP with features: [pred_t, actual_{t-1}, rolling_std_proxy]
    feat_int = np.column_stack([
        proposed_preds[:test_size - 1],
        actual_test[:test_size - 1],
        np.full(test_size - 1, roll_std),
    ]).astype(np.float32)
    tgt_high = (actual_test[1:test_size] + roll_std).astype(np.float32)
    tgt_low  = (actual_test[1:test_size] - roll_std).astype(np.float32)

    sc_int   = StandardScaler()
    feat_s   = sc_int.fit_transform(feat_int)
    int_tr   = int(len(feat_s) * 0.7)
    int_m    = train_interval_mlp(
        feat_s[:int_tr],
        np.column_stack([tgt_high[:int_tr], tgt_low[:int_tr]])
    )
    int_m.eval()
    with torch.no_grad():
        int_out = int_m(torch.FloatTensor(feat_s)).numpy()
    pred_high_raw = int_out[:, 0]
    pred_low_raw  = int_out[:, 1]
    pred_high_raw = np.maximum(pred_high_raw, pred_low_raw)  # enforce ≥ low

    # Pad first point
    ph = np.concatenate([[proposed_preds[0] + roll_std], pred_high_raw])[:test_size]
    pl = np.concatenate([[proposed_preds[0] - roll_std], pred_low_raw])[:test_size]
    print("done")

    # ── 7. Trading ────────────────────────────────────────────
    print("  [6/6] Trading simulation...", end=' ', flush=True)
    cum_b, sh_b = scheme1_basic(actual_test, proposed_preds, cost=BASE_COST)
    cum_i, sh_i = scheme2_interval(actual_test, proposed_preds, ph, pl, cost=BASE_COST)

    cost_res = {}
    for c in COST_LEVELS:
        cr, sr = scheme2_interval(actual_test, proposed_preds, ph, pl, cost=c)
        cost_res[c] = (float(cr[-1] * 100), float(sr))

    # Robustness: per-window MAPE
    rob_res = {}
    for W in ROB_WINDOWS:
        if test_size >= W:
            rob_res[W] = float(
                mean_absolute_percentage_error(actual_test[:W],
                                               proposed_preds[:W]) * 100)
        else:
            rob_res[W] = float('nan')

    drawdown = (np.maximum.accumulate(cum_i + 1) - (cum_i + 1))
    print("done")

    # ── store ─────────────────────────────────────────────────
    results[stock_name] = {
        'metrics':      met_results,
        'dm':           dm_results,
        'true':         actual_test,
        'preds':        model_preds,
        'dates_test':   dates_test,
        'cum_basic':    cum_b,
        'cum_int':      cum_i,
        'sharpe_basic': float(sh_b),
        'sharpe_int':   float(sh_i),
        'cost_res':     cost_res,
        'rob_res':      rob_res,
        'imf_selected': imf_selected,
        't5_rows':      t5_rows,
        'complexities': complexities,
        'u_matrix':     u_matrix,
        'series':       series,
        'pred_high':    ph,
        'pred_low':     pl,
        'drawdown':     drawdown,
    }
    r, m, mpe = met_results['Proposed']
    print(f"  [OK] Proposed  RMSE={r:.4f}  MAE={m:.4f}  MAPE={mpe:.2f}%")
    print(f"    Trading  Basic Sharpe={sh_b:.3f}  Interval Sharpe={sh_i:.3f}")



# ═══════════════════════════════════════════════════════════════
# TABLE GENERATION
# ═══════════════════════════════════════════════════════════════
def _wtex(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def generate_tables(results, df_merged, avail_ind):
    snames = [n for _, n in STOCKS]

    # ── Table 4: Descriptive statistics per stock ─────────────
    for idx, (col, name) in enumerate(STOCKS, 1):
        if col not in df_merged.columns: continue
        s = df_merged[col].dropna().values
        row = pd.DataFrame([{
            'Mean': s.mean(), 'Max': s.max(), 'Min': s.min(),
            'Std': s.std(), 'Obs': len(s)
        }])
        _wtex(os.path.join(RES_DIR, f'Table_4_{idx}.tex'),
              row.to_latex(index=False, float_format='%.4f'))

    # ── Table 2: Predictor stats ──────────────────────────────
    ind_label = {
        'SP500': 'S\\&P 500', 'Shanghai_Index': 'Shanghai Index',
        'Crude_Oil': 'Crude Oil', 'US_Dollar_Index': 'US Dollar Index',
        'VIX': 'VIX', 'Search_Index': 'Search Index',
        'News_Sentiment': 'News Sentiment'
    }
    tex = ("\\begin{tabular}{lrrrrr} \\toprule\n"
           "Predictor & Mean & Max & Min & Std & Obs \\\\ \\midrule\n")
    for col in avail_ind:
        s = df_merged[col].dropna().values
        label = ind_label.get(col, col.replace('_', '\\_'))
        tex += (f"{label} & {s.mean():.4f} & {s.max():.4f} & "
                f"{s.min():.4f} & {s.std():.4f} & {len(s)} \\\\ \n")
    tex += "\\bottomrule \\end{tabular}"
    _wtex(os.path.join(RES_DIR, 'Table_2_Predictor_Stats.tex'), tex)

    # ── Table 5: VMD mode statistics ─────────────────────────
    for idx, name in enumerate(snames, 1):
        if name not in results: continue
        rows = results[name]['t5_rows']
        tex  = ("{\\tiny\\setlength{\\tabcolsep}{1pt}"
                "\\begin{tabular}{lrrrrrr} \\toprule\n"
                "Mode & Freq & Per & VarR & Corr & ApEn & Type"
                " \\\\ \\midrule\n")
        for r in rows:
            tex += (f"IMF{r['Mode'][-1]} & {r['Freq']:.3f} & "
                    f"{r['Period']:.1f} & {r['VarR']:.3f} & "
                    f"{r['Corr']:.3f} & {r['ApEn']:.3f} & "
                    f"{r['Type']} \\\\ \n")
        tex += "\\bottomrule \\end{tabular}}"
        _wtex(os.path.join(RES_DIR, f'Table_5_{idx}.tex'), tex)

    # ── Table 6: LASSO selected features per IMF ─────────────
    ind_short = {
        'SP500': 'S\\&P', 'Shanghai_Index': 'Shg',
        'Crude_Oil': 'Cru', 'US_Dollar_Index': 'USD',
        'VIX': 'VIX', 'Search_Index': 'Src', 'News_Sentiment': 'Nws'
    }
    hist_orig  = [f'target_lag{i}' for i in range(1, LAG + 1)]
    hist_short = [f'h{i}' for i in range(1, LAG + 1)]
    feat_orig  = [c for c in avail_ind] + hist_orig
    feat_short = [ind_short.get(c, c[:3]) for c in avail_ind] + hist_short

    for idx, name in enumerate(snames, 1):
        if name not in results: continue
        imf_sel = results[name]['imf_selected']
        ncols   = len(feat_short)
        tex  = ("{\\tiny\\setlength{\\tabcolsep}{1pt}"
                f"\\begin{{tabular}}{{l{'l'*ncols}}} \\toprule\n"
                f"IMF & {' & '.join(feat_short)} \\\\ \\midrule\n")
        for imf, sel in imf_sel.items():
            row_tex = f"IMF{imf[-1]} "
            for fo in feat_orig:
                row_tex += "& X " if fo in sel else "& "
            tex += row_tex + "\\\\ \n"
        tex += "\\bottomrule \\end{tabular}}"
        _wtex(os.path.join(RES_DIR, f'Table_6_{idx}.tex'), tex)

    # ── Table 7: Forecast accuracy ────────────────────────────
    for idx, name in enumerate(snames, 1):
        if name not in results: continue
        met = results[name]['metrics']
        tex = ("{\\tiny\\setlength{\\tabcolsep}{2pt}"
               "\\begin{tabular}{lrrr} \\toprule\n"
               "Model & RMSE & MAE & MAPE \\\\ \\midrule\n")
        for mn in BENCH_NAMES:
            if mn not in met: continue
            rm, ma, mp = met[mn]
            tex += (f"{mn.replace('_', '-')} & {rm:.4f} & "
                    f"{ma:.4f} & {mp:.2f}\\% \\\\ \n")
        tex += "\\bottomrule \\end{tabular}}"
        _wtex(os.path.join(RES_DIR, f'Table_7_{idx}.tex'), tex)

    # ── Table 8: Trading performance ─────────────────────────
    for idx, name in enumerate(snames, 1):
        if name not in results: continue
        r   = results[name]
        ret_b = float(r['cum_basic'][-1] * 100) if len(r['cum_basic']) else 0
        ret_i = float(r['cum_int'][-1]   * 100) if len(r['cum_int'])   else 0
        tex = ("{\\tiny\\setlength{\\tabcolsep}{2pt}"
               "\\begin{tabular}{lrr} \\toprule\n"
               "Strategy & Ret(\\%) & Sharpe \\\\ \\midrule\n"
               f"Basic & {ret_b:.2f} & {r['sharpe_basic']:.3f} \\\\ \n"
               f"Interval & {ret_i:.2f} & {r['sharpe_int']:.3f} \\\\ \n"
               "\\bottomrule \\end{tabular}}")
        _wtex(os.path.join(RES_DIR, f'Table_8_{idx}.tex'), tex)

    # ── Table 9: Transaction cost sensitivity ─────────────────
    tex = ("\\begin{tabular}{lrrr} \\toprule\n"
           "Stock & 0.0\\% & 0.1\\% & 0.2\\% \\\\ \\midrule\n")
    for name in snames:
        if name not in results: continue
        cr = results[name]['cost_res']
        r0  = cr.get(0.000, (0, 0))[0]
        r01 = cr.get(0.001, (0, 0))[0]
        r02 = cr.get(0.002, (0, 0))[0]
        tex += f"{name} & {r0:.2f}\\% & {r01:.2f}\\% & {r02:.2f}\\% \\\\ \n"
    tex += "\\bottomrule \\end{tabular}"
    _wtex(os.path.join(RES_DIR, 'Table_9_Transaction_Cost.tex'), tex)

    # ── Table 10: Robustness (sliding window MAPE) ────────────
    for idx, name in enumerate(snames, 1):
        if name not in results: continue
        rob = results[name]['rob_res']
        tex = ("{\\tiny\\setlength{\\tabcolsep}{2pt}"
               "\\begin{tabular}{lr} \\toprule\n"
               "Window & MAPE(\\%) \\\\ \\midrule\n")
        for W, val in rob.items():
            tex += (f"W={W} & {val:.2f}\\% \\\\ \n"
                    if not np.isnan(val) else f"W={W} & --- \\\\ \n")
        tex += "\\bottomrule \\end{tabular}}"
        _wtex(os.path.join(RES_DIR, f'Table_10_{idx}.tex'), tex)
    # also write the single-version for Table_10_Robustness_Check.tex
    # (used by the older LaTeX template)
    rob_all = {n: results[n]['rob_res'] for n in snames if n in results}
    tex_rob = ("\\begin{tabular}{llr} \\toprule\n"
               "Stock & Window & MAPE(\\%) \\\\ \\midrule\n")
    for name in snames:
        if name not in rob_all: continue
        for W, val in rob_all[name].items():
            tex_rob += (f"{name} & W={W} & {val:.2f}\\% \\\\ \n"
                        if not np.isnan(val) else
                        f"{name} & W={W} & --- \\\\ \n")
    tex_rob += "\\bottomrule \\end{tabular}"
    _wtex(os.path.join(RES_DIR, 'Table_10_Robustness_Check.tex'), tex_rob)

    # ── Table 11: Diebold-Mariano test ────────────────────────
    tex = ("\\begin{tabular}{llrrl} \\toprule\n"
           "Stock & Benchmark & DM Stat & p-value & Sig \\\\ \\midrule\n")
    for name in snames:
        if name not in results: continue
        for bname, (stat, pval) in results[name]['dm'].items():
            sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "ns")
            tex += (f"{name} & {bname} & {stat:.3f} & "
                    f"{pval:.4f} & {sig} \\\\ \n")
    tex += "\\bottomrule \\end{tabular}\n\\footnotesize{*p<0.05, **p<0.01}"
    _wtex(os.path.join(RES_DIR, 'Table_11_DM_Test.tex'), tex)

    # ── Table 12: Model hyperparameters ──────────────────────
    params = [
        ('VMD Modes (K)', K_MD), ('VMD Alpha', ALPHA_VMD),
        ('Lag Window', LAG), ('LSTM Hidden', LSTM_HIDDEN),
        ('LSTM Epochs', LSTM_EPOCHS), ('ELM Hidden', ELM_HIDDEN),
        ('LASSO CV Folds', 5), ('Ridge Alpha (ARX)', RIDGE_ALPHA),
        ('Train Ratio', f'{TRAIN_RATIO:.0%}'),
        ('Base Tx Cost', f'{BASE_COST:.2%}'),
    ]
    tex = ("\\begin{tabular}{lr} \\toprule\n"
           "Parameter & Value \\\\ \\midrule\n")
    for k, v in params:
        tex += f"{k} & {v} \\\\ \n"
    tex += "\\bottomrule \\end{tabular}"
    _wtex(os.path.join(RES_DIR, 'Table_12_Parameters.tex'), tex)

    # ── Table 1: Data description (static) ───────────────────
    t1 = pd.DataFrame([
        {'Variable': 'Target\\_Close', 'Description': 'Individual Rare Earth Company Stock Close'},
        {'Variable': 'SP500',          'Description': 'S\\&P 500 Index'},
        {'Variable': 'Shanghai\\_Index','Description': 'Shanghai Composite'},
        {'Variable': 'Crude\\_Oil',    'Description': 'WTI Crude Oil Futures'},
        {'Variable': 'USD\\_CNY',      'Description': 'USD/CNY Exchange Rate'},
        {'Variable': 'Search\\_Index', 'Description': 'Google Trends (worldwide)'},
        {'Variable': 'News\\_Sentiment','Description': 'NESI Score (Fed SF)'},
    ])
    _wtex(os.path.join(RES_DIR, 'Table_1_Data_Description.tex'),
          t1.to_latex(index=False))

    print("  [OK] All tables written")


# ═══════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════
def _savefig(fig, name, row):
    path = os.path.join(RES_DIR, f'{name}_row{row}.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def generate_figures(results, df_merged):
    pairs = [(STOCKS[i*2], STOCKS[i*2+1]) for i in range(4)]

    for row_i, ((c1, n1), (c2, n2)) in enumerate(pairs, 1):
        r1 = results.get(n1, {})
        r2 = results.get(n2, {})

        # ── Fig 2: Full price series ──────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, col, name in zip(axes, [c1, c2], [n1, n2]):
            if col in df_merged.columns:
                ax.plot(df_merged['Date'], df_merged[col].ffill().values,
                        color='steelblue', linewidth=0.7)
            ax.set_title(name); ax.set_xlabel('Date')
            ax.set_ylabel('Price'); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _savefig(fig, 'Figure_2_Original_Price_Series', row_i)

        # ── Fig 3: VMD decomposition ──────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, name in zip(axes, [n1, n2]):
            if name in results:
                u = results[name]['u_matrix']
                for i in range(K_MD):
                    shift = i * np.ptp(u[i]) * 1.6
                    ax.plot(u[i] - shift, linewidth=0.6,
                            label=f'IMF{i+1} ({results[name]["complexities"][i][0]})')
                ax.set_title(name); ax.legend(fontsize=7, ncol=2)
                ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _savefig(fig, 'Figure_3_VMD_Decomposition', row_i)

        # ── Fig 4: Zoomed price ───────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, col, name in zip(axes, [c1, c2], [n1, n2]):
            if col in df_merged.columns:
                vals = df_merged[col].ffill().dropna().values
                ax.plot(vals[-120:], color='darkorange', linewidth=0.8)
            ax.set_title(f'{name} — Last 120 Days')
            ax.set_xlabel('Days'); ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _savefig(fig, 'Figure_4_Zoomed_Price', row_i)

        # ── Fig 5: Correlation heatmap ────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
        for ax, col, name in zip(axes, [c1, c2], [n1, n2]):
            num_cols = [cc for cc in [col] + list(df_merged.select_dtypes('number').columns)
                        if cc in df_merged.columns][:8]
            corr = df_merged[num_cols].corr()
            sns.heatmap(corr, ax=ax, annot=True, fmt='.2f',
                        cmap='coolwarm', cbar=False, annot_kws={'size': 7})
            ax.set_title(name); ax.tick_params(labelsize=7)
        fig.tight_layout()
        _savefig(fig, 'Figure_5_Feature_Heatmap', row_i)

        # ── Fig 6: Predictions (all 5 models) ─────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        colors = {'Proposed': ('red', '--'), 'LSTM': ('blue', ':'),
                  'VMD_LSTM': ('green', '-.'), 'BP': ('purple', ':'),
                  'ELM': ('orange', ':')}
        for ax, name in zip(axes, [n1, n2]):
            if name not in results: continue
            dts  = results[name]['dates_test']
            true = results[name]['true']
            n_sh = min(len(true), 150)
            ax.plot(dts[-n_sh:], true[-n_sh:],
                    color='black', linewidth=1.2, label='Actual')
            for mname, (col, ls) in colors.items():
                if mname in results[name]['preds']:
                    p = results[name]['preds'][mname]
                    ax.plot(dts[-n_sh:], p[-n_sh:],
                            color=col, linestyle=ls, linewidth=0.8,
                            label=mname)
            ax.set_title(name); ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _savefig(fig, 'Figure_6_Prediction_Plot', row_i)

        # ── Fig 8: ApEn bar chart per IMF ─────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        for ax, name in zip(axes, [n1, n2]):
            if name not in results: continue
            t5  = results[name]['t5_rows']
            aps = [r['ApEn'] for r in t5]
            cols = ['salmon' if r['Type'] == 'High' else 'teal' for r in t5]
            ax.bar([f'IMF{i+1}' for i in range(K_MD)], aps, color=cols)
            ax.axhline(results[name].get('orig_apen', 0),
                       color='black', linestyle='--', linewidth=1,
                       label='Orig ApEn')
            ax.set_title(f'{name} (red=High, teal=Low)')
            ax.set_xlabel('Mode'); ax.set_ylabel('ApEn')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _savefig(fig, 'Figure_8_Entropy', row_i)

        # ── Figs 9 & 10: Equity curves ────────────────────────
        for fig_name, key, clr, strat in [
            ('Figure_9_Cumulative_Return_Basic',     'cum_basic', 'steelblue', 'Basic'),
            ('Figure_10_Cumulative_Return_Interval', 'cum_int',   'green',     'Interval'),
        ]:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            for ax, name in zip(axes, [n1, n2]):
                if name not in results: continue
                cum = results[name].get(key, [])
                if len(cum) == 0: continue
                ax.plot(cum * 100, color=clr, linewidth=1)
                ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
                ax.fill_between(range(len(cum)),
                                np.where(cum * 100 >= 0, cum * 100, 0), 0,
                                color=clr, alpha=0.15)
                ax.set_title(f'{name} — {strat}')
                ax.set_xlabel('Trading Days'); ax.set_ylabel('Return (%)')
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _savefig(fig, fig_name, row_i)

        # ── Fig 11: Drawdown ──────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        for ax, name in zip(axes, [n1, n2]):
            if name not in results: continue
            dd = results[name].get('drawdown', [])
            if len(dd) == 0: continue
            ax.fill_between(range(len(dd)), -dd * 100, 0,
                            color='crimson', alpha=0.45)
            ax.plot(-dd * 100, color='darkred', linewidth=0.8)
            ax.set_title(f'{name} — Max Drawdown')
            ax.set_xlabel('Trading Days'); ax.set_ylabel('Drawdown (%)')
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _savefig(fig, 'Figure_11_Drawdown', row_i)

        # ── Fig 13: Robustness check ──────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        for ax, name in zip(axes, [n1, n2]):
            if name not in results: continue
            rob = results[name]['rob_res']
            Ws  = list(rob.keys())
            mps = [rob[W] for W in Ws]
            Ws_clean  = [W for W, v in zip(Ws, mps) if not np.isnan(v)]
            mps_clean = [v for v in mps if not np.isnan(v)]
            if Ws_clean:
                ax.plot(Ws_clean, mps_clean, marker='o', color='saddlebrown',
                        linewidth=1.2)
            ax.set_title(f'{name} — MAPE by window')
            ax.set_xlabel('Window Size (days)'); ax.set_ylabel('MAPE (%)')
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _savefig(fig, 'Figure_13_Robustness_Check', row_i)

    print("  [OK] All figures written")


# ═══════════════════════════════════════════════════════════════
# README
# ═══════════════════════════════════════════════════════════════
README_TEXT = """
README — Rare Earth Minerals Forecasting Pipeline
==================================================

OVERVIEW
--------
Implements the VMD-ARX-LSTM hybrid paper framework across 8 rare earth stocks.
Produces all paper Tables (4-12) and Figures (2-13) with real computed values.

MODELS
------
  BP        : Backpropagation MLP (2 hidden layers)
  ELM       : Extreme Learning Machine (random projection + ridge output)
  LSTM      : Standalone LSTM on raw price + indicators (no decomposition)
  VMD_LSTM  : VMD decomposition (K=6) + LSTM on every IMF  [ablation]
  Proposed  : VMD + ApEn classification + LASSO selection
              -> ARX (Ridge) for Low-complexity IMFs
              -> LSTM for High-complexity IMFs          [paper contribution]

DATA FILES (Data/)
------------------
  Top Rare Earth Mineral Companies and the Stock Price.xlsx
    Sheet "Stock Price" — 8 company daily closing prices
  aligned_dataset.csv
    Indicators: SP500, Shanghai_Index, Crude_Oil, USD_CNY,
                Search_Index, News_Sentiment + averaged Target_Close
  news_sentiment_data.xlsx — NESI Federal Reserve SF sentiment scores
  rare_earth_trends.csv    — Google Trends "rare earth" keyword

STEP-BY-STEP: HOW TO RUN
-------------------------
Step 1 — Install dependencies:
    pip install pandas numpy scikit-learn torch vmdpy antropy seaborn
               matplotlib openpyxl scipy

Step 2 — (Optional) Inject sentiment data:
    python inject_sentiment.py
    Merges news_sentiment_data.xlsx into aligned_dataset.csv

Step 3 — Run the FULL pipeline (all models, all stocks):
    python run_all.py
    • Trains BP, ELM, LSTM, VMD-LSTM, Proposed on each of 8 stocks
    • Computes DM test, trading strategies, robustness check
    • Generates Tables 4-12 (.tex) and Figures 2-13 (.png) in Results/
    • Runtime: ~20-40 min on CPU (LSTM_EPOCHS=50 default)
      Set LSTM_EPOCHS=100 in run_all.py for paper-exact results.

Step 4 — Compile the LaTeX report:
    cd Results
    pdflatex rare_earth_report.tex

OUTPUTS (Results/)
------------------
  all_forecasts.csv       — All 5 model predictions for every stock
  Table_4_{1-8}.tex       — Descriptive statistics per stock
  Table_5_{1-8}.tex       — VMD mode info (frequency, period, ApEn)
  Table_6_{1-8}.tex       — LASSO selected features per IMF per stock
  Table_7_{1-8}.tex       — RMSE / MAE / MAPE for 5 models
  Table_8_{1-8}.tex       — Trading returns and Sharpe ratios
  Table_9_Transaction_Cost.tex
  Table_10_{1-8}.tex      — Robustness: MAPE at different window sizes
  Table_11_DM_Test.tex    — Diebold-Mariano test vs each benchmark
  Table_12_Parameters.tex — Hyperparameters
  Figure_2_*.png          — Price series  (4 rows × 1x2 grid)
  Figure_3_*.png          — VMD modes
  Figure_4_*.png          — Zoomed price
  Figure_5_*.png          — Correlation heatmaps
  Figure_6_*.png          — Forecast vs actual (all 5 models)
  Figure_8_*.png          — ApEn per IMF
  Figure_9_*.png          — Equity curve (basic strategy)
  Figure_10_*.png         — Equity curve (interval strategy)
  Figure_11_*.png         — Max drawdown
  Figure_13_*.png         — Robustness check

KEY PARAMETERS
--------------
  VMD modes (K)    : 6
  VMD alpha        : 3000
  Lag window       : 5 days
  Train/test split : 80/20
  LSTM hidden      : 64 units
  LSTM epochs      : 50 (set 100 in run_all.py for paper-exact)
  ELM hidden       : 500 units
  LASSO CV folds   : 5
  Base tx cost     : 0.05%

DM TEST SIGNIFICANCE
--------------------
  ns = not significant
  *  = significant at 5%
  ** = significant at 1%

GITHUB
------
  Upload entire Rare_Earth_Minerals/ folder (excluding Data/ large files)
  Include: Code/, Results/, run_all.py, readme.txt, requirements.txt
""".strip()


def write_readme():
    path = os.path.join(BASE_DIR, 'readme.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(README_TEXT)
    print("  [OK] readme.txt written")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  RARE EARTH MINERALS — FULL PAPER PIPELINE")
    print(f"  Models: {', '.join(BENCH_NAMES)}")
    print(f"  Stocks: {len(STOCKS)}  |  VMD K={K_MD}  |  Epochs={LSTM_EPOCHS}")
    print("=" * 65)

    print("\n[0] Loading data...")
    df_merged, avail_ind = load_stock_data()
    print(f"    Merged shape: {df_merged.shape}")
    print(f"    Indicators : {avail_ind}")

    results = {}
    for stock_col, stock_name in STOCKS:
        if stock_col not in df_merged.columns:
            print(f"\n  WARNING: '{stock_col}' not in dataframe — skipping.")
            continue
        try:
            run_stock(stock_col, stock_name, df_merged, avail_ind, results)
        except Exception as exc:
            import traceback
            print(f"\n  ERROR processing {stock_name}: {exc}")
            traceback.print_exc()

    print(f"\n\n[2] Generating tables ({len(results)} stocks)...")
    generate_tables(results, df_merged, avail_ind)

    print("\n[3] Generating figures...")
    generate_figures(results, df_merged)

    print("\n[4] Writing readme.txt...")
    write_readme()

    # Consolidated forecast CSV
    rows = []
    for name, r in results.items():
        for i, d in enumerate(r['dates_test']):
            row = {'Stock': name, 'Date': d,
                   'Actual': float(r['true'][i])}
            for mn in BENCH_NAMES:
                if mn in r['preds'] and i < len(r['preds'][mn]):
                    row[mn] = float(r['preds'][mn][i])
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(RES_DIR, 'all_forecasts.csv'), index=False)
        print("  [OK] all_forecasts.csv saved")

    # Summary table to stdout
    print("\n" + "=" * 65)
    print(f"{'Stock':<18} {'RMSE':>9} {'MAE':>9} {'MAPE%':>8} "
          f"{'IntRet%':>8} {'IntSharpe':>10}")
    print("-" * 65)
    for (_, name) in STOCKS:
        if name not in results: continue
        rm, ma, mp = results[name]['metrics']['Proposed']
        cr = results[name]['cum_int']
        ret = float(cr[-1] * 100) if len(cr) else 0
        sh  = results[name]['sharpe_int']
        print(f"{name:<18} {rm:>9.4f} {ma:>9.4f} {mp:>8.2f} "
              f"{ret:>8.2f} {sh:>10.3f}")
    print("=" * 65)
    print("\n[DONE] All results saved to Results/")


if __name__ == '__main__':
    main()
