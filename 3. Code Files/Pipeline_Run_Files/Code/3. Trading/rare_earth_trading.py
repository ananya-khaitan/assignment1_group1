import pandas as pd
import numpy as np
import os

def scheme1_basic(df, cost=0.0):
    returns = []
    
    for i in range(1, len(df)):
        actual_t_minus_1 = df['Actual'].iloc[i-1]
        pred_t = df['Predicted'].iloc[i]
        actual_t = df['Actual'].iloc[i]
        
        # Predict Return
        pred_return = (pred_t - actual_t_minus_1) / actual_t_minus_1
        true_return = (actual_t - actual_t_minus_1) / actual_t_minus_1
        
        if pred_return > cost:
            position = 1   # Long
        elif pred_return < -cost:
            position = -1  # Short
        else:
            position = 0   # Hold
            
        returns.append(position * true_return)
        
    returns = np.array(returns)
    cum_ret = np.cumprod(1 + returns) - 1
    sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
    
    return cum_ret[-1], sharpe

def scheme2_interval_constrained(df, cost=0.0005):
    returns = []
    
    for i in range(1, len(df)):
        actual_t_minus_1 = df['Actual'].iloc[i-1]
        pred_t = df['Predicted'].iloc[i]
        actual_t = df['Actual'].iloc[i]
        
        pred_high = df['Pred_High'].iloc[i]
        pred_low = df['Pred_Low'].iloc[i]
        
        # Predict Return
        pred_return = (pred_t - actual_t_minus_1) / actual_t_minus_1
        true_return = (actual_t - actual_t_minus_1) / actual_t_minus_1
        
        # Interval Constraint: Point prediction MUST fall within predicted High/Low bounds
        is_confident = (pred_t >= pred_low) and (pred_t <= pred_high)
        
        if is_confident:
            if pred_return > cost:
                position = 1
            elif pred_return < -cost:
                position = -1
            else:
                position = 0
        else:
            position = 0 # No trade due to high uncertainty
            
        returns.append(position * true_return)
        
    returns = np.array(returns)
    cum_ret = np.cumprod(1 + returns) - 1
    sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
    
    return cum_ret[-1], sharpe

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, 'Results')
    input_file = os.path.join(results_dir, 'interval_forecasts.csv')
    output_file = os.path.join(results_dir, 'trading_performance.csv')

    print("Running Trading Strategy Simulations on out-of-sample data...")
    df = pd.read_csv(input_file)
    
    # 0.05% Transaction Cost
    cost = 0.0005

    ret_s1, sh_s1 = scheme1_basic(df, cost=0.0)
    ret_s2_int, sh_s2_int = scheme2_interval_constrained(df, cost=cost)
    
    results = pd.DataFrame([{
        'Strategy': "Scheme 1 (Naive Basic Directional)",
        'Cumulative Return (%)': ret_s1 * 100,
        'Annualized Sharpe Ratio': sh_s1
    }, {
        'Strategy': "Scheme 2' (Proposed Interval-Constrained with 0.05% Cost)",
        'Cumulative Return (%)': ret_s2_int * 100,
        'Annualized Sharpe Ratio': sh_s2_int
    }])
    
    print("\n--- FINAL TRADING PERFORMANCE ---")
    print(results.to_string(index=False))
    print("---------------------------------")
    
    results.to_csv(output_file, index=False)
    print(f"Trading metrics saved to {output_file}")

if __name__ == '__main__':
    main()
