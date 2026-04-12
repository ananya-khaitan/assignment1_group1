import pandas as pd
import numpy as np
import os
import antropy as ant
from vmdpy import VMD
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import pickle

def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df

def create_lags(df, lags=5):
    """Create lagged features for all columns up to 'lags' days."""
    lagged_df = df.copy()
    feature_cols = []
    
    for col in df.columns:
        for lag in range(1, lags + 1):
            lag_col_name = f'{col}_lag{lag}'
            lagged_df[lag_col_name] = df[col].shift(lag)
            feature_cols.append(lag_col_name)
            
    lagged_df.dropna(inplace=True)
    return lagged_df, feature_cols

def approximate_entropy(series):
    # Using antropy for fast Approximate Entropy calculation
    return ant.app_entropy(series, order=2)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'Data')
    input_file = os.path.join(data_dir, 'aligned_dataset.csv')
    output_file = os.path.join(data_dir, 'processed_imfs_features.pkl')

    print("Loading preprocessed dataset...")
    df = load_data(input_file)
    
    # 1. Create Lagged Features
    print("Creating lagged features (5 days)...")
    lagged_df, candidate_features = create_lags(df, lags=5)
    
    # Ensure even length for VMD
    if len(lagged_df) % 2 != 0:
        lagged_df = lagged_df.iloc[:-1]
        
    target = lagged_df['Target_Close'].values

    # 2. VMD Decomposition
    print("Running VMD Decomposition (K=4)...")
    K = 4
    # alpha=2000, tau=0, K=4, DC=0, init=1, tol=1e-7
    u, u_hat, omega = VMD(target, 2000, 0, K, 0, 1, 1e-7)
    
    # 3. Approximate Entropy (Complexity Assessment)
    print("Computing Approximate Entropy for Original Series vs IMFs...")
    orig_apen = approximate_entropy(target)
    print(f"Original Series ApEn: {orig_apen:.4f}")
    
    imf_complexities = {}
    for i in range(K):
        imf = u[i, :]
        apen = approximate_entropy(imf)
        complexity = 'High' if apen > orig_apen else 'Low'
        imf_complexities[f'IMF_{i+1}'] = {'ApEn': apen, 'Complexity': complexity}
        print(f"IMF {i+1} ApEn: {apen:.4f} -> {complexity} Complexity")

    # 4. Adaptive LASSO Feature Selection per IMF
    print("\nRunning LASSO Feature Selection for each IMF...")
    imf_features = {}
    
    X = lagged_df[candidate_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for i in range(K):
        imf_name = f'IMF_{i+1}'
        imf_target = u[i, :len(X_scaled)] # Ensure safe length matching
        
        # Fit LassoCV
        lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
        lasso.fit(X_scaled, imf_target)
        
        # Extract non-zero features
        coefs = pd.Series(lasso.coef_, index=candidate_features)
        selected = coefs[coefs != 0].index.tolist()
        
        imf_features[imf_name] = {
            'target': imf_target,
            'complexity': imf_complexities[imf_name]['Complexity'],
            'features': selected,
            'alpha': lasso.alpha_
        }
        print(f"[{imf_name}] Selected {len(selected)} features out of {len(candidate_features)}")

    # 5. Save Results
    output_data = {
        'dates': lagged_df.index,
        'lagged_df': lagged_df,
        'imfs': imf_features,
        'original_target': target,
        'target_high': lagged_df['Target_High'].values,
        'target_low': lagged_df['Target_Low'].values
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
        
    print(f"\nSuccessfully completed Decomposition and Feature Selection.")
    print(f"Saved data to {output_file}")

if __name__ == '__main__':
    main()
