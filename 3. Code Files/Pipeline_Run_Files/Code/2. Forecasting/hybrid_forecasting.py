import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class FastLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_lstm(X_train, y_train, epochs=100, lr=0.01):
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1) # Add seq len 1
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)

    model = FastLSTM(X_train.shape[1], hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
    return model

def predict_lstm(model, X_test):
    model.eval()
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
    with torch.no_grad():
        preds = model(X_test_t)
    return preds.numpy().flatten()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'Data')
    results_dir = os.path.join(base_dir, 'Results')
    
    input_file = os.path.join(data_dir, 'processed_imfs_features.pkl')
    output_file = os.path.join(results_dir, 'point_forecasts.csv')

    print("Loading extracted IMFs and LASSO features...")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    imfs = data['imfs']
    lagged_df = data['lagged_df']
    dates = data['dates'].values
    actual_target = data['original_target']
    
    n_samples = len(actual_target)
    train_size = int(n_samples * 0.8)
    
    test_dates = dates[train_size:]
    test_actual = actual_target[train_size:]
    test_high = data['target_high'][train_size:]
    test_low = data['target_low'][train_size:]
    
    imf_test_preds = np.zeros((len(imfs), n_samples - train_size))

    print(f"\nData split: {train_size} Train | {n_samples - train_size} Test")
    
    for idx, (imf_name, info) in enumerate(imfs.items()):
        y = info['target']
        comp = info['complexity']
        features = info['features']
        
        print(f"\n--- Processing {imf_name} [{comp} Complexity] ---")
        
        # Build X matrix with ONLY the selected LASSO features
        if len(features) == 0:
            print("No features selected by LASSO, falling back to autoregressive lag1")
            features = ['Target_Close_lag1'] # Fallback
            
        X = lagged_df[features].values
        
        # Scale X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        if comp == 'Low':
            print("Training ARX Linear model (Ridge Regression)...")
            try:
                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            except Exception as e:
                print(f"Linear model failed: {e}. Falling back to 0 prediction.")
                preds = np.zeros(len(X_test))
                
        elif comp == 'High':
            print("Training LSTM neural network...")
            model = train_lstm(X_train, y_train, epochs=20, lr=0.005) # reduced epochs for demo speed
            preds = predict_lstm(model, X_test)
            
        imf_test_preds[idx, :] = preds

    # Final Ensemble (Linear Addition)
    print("\n--- Final Ensemble ---")
    final_preds = np.sum(imf_test_preds, axis=0)
    
    # Calculate Metrics
    rmse = np.sqrt(mean_squared_error(test_actual, final_preds))
    mae = mean_absolute_error(test_actual, final_preds)
    mape = mean_absolute_percentage_error(test_actual, final_preds) * 100
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test MAPE: {mape:.4f}%")
    
    # Save the predictions to be used by Trading strategy and Interval models
    df_res = pd.DataFrame({
        'Date': test_dates,
        'Actual': test_actual,
        'Predicted': final_preds,
        'High': test_high,
        'Low': test_low
    })
    
    df_res.to_csv(output_file, index=False)
    print(f"\nSaved point forecasts to {output_file}")

if __name__ == '__main__':
    main()
