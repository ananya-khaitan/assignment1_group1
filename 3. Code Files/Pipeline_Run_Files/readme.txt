README — Rare Earth Minerals Forecasting Pipeline
==================================================

OVERVIEW
--------
Implements the VMD-ARX-LSTM hybrid paper framework across 8 rare earth stocks.
Produces all paper Tables (4-12) and Figures (2-13) with real computed values.

MODELS
------
  Proposed  : VMD + ApEn classification + LASSO selection
              -> ARX (Ridge) for Low-complexity IMFs
              -> LSTM for High-complexity IMFs
  
  Deep Learning Benchmarks:
  LSTM      : Standalone LSTM on raw price + indicators (no decomposition)
  VMD_LSTM  : VMD decomposition (K=6) + LSTM on every IMF
  BP / MLP  : Backpropagation Multi-Layer Perceptron
  RNN       : Simple Recurrent Neural Network
  
  Machine Learning Benchmarks:
  ELM       : Extreme Learning Machine 
  SVR       : Support Vector Regression (or SVM for classification)
  RF        : Random Forest 
  ExtraTrees: Extremely Randomized Trees
  Lasso     : L1-regularized model
  NB        : Naïve Bayes (Classification Baseline)
  
  Statistical Benchmarks:
  ARIMA     : Autoregressive Integrated Moving Average
  ES        : Exponential Smoothing

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