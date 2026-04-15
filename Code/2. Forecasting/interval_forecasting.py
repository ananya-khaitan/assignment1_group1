import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class IntervalMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Output: High, Low
        )

    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, y_train, epochs=200, lr=0.005):
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)

    model = IntervalMLP(X_train.shape[1])
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

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, 'Results')
    
    input_file = os.path.join(results_dir, 'point_forecasts.csv')
    output_file = os.path.join(results_dir, 'interval_forecasts.csv')

    print("Loading point forecasts and interval targets...")
    df = pd.read_csv(input_file)
    
    # We will build a dataset to predict High_t and Low_t
    # Input features: Predicted_t, High_{t-1}, Low_{t-1}, Actual_{t-1}
    
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Actual'] = df['Actual'].shift(1)
    
    df.dropna(inplace=True)
    
    features = ['Predicted', 'Prev_High', 'Prev_Low', 'Prev_Actual']
    targets = ['High', 'Low']
    
    X = df[features].values
    y = df[targets].values
    
    # Train/Test split: 80/20 of this *testing* partition (or re-split)
    # The paper trains the interval model. We can just use the first 80% of df
    n_samples = len(df)
    train_size = int(n_samples * 0.8)
    
    X_train, X_test = X[:, :], X[train_size:, :] # Wait, we must predict ALL of test_set for the strategy.
    # To properly simulate, let's just train on the first 50% and test on the last 50% of these test values?
    # Or we can just build a fast model trained on 80% of this partition.
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    
    print(f"Training Interval MLP on {train_size} samples...")
    model = train_mlp(X_train_s, y_train, epochs=150)
    
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test_s)).numpy()
        
    df_res = df.iloc[train_size:].copy()
    df_res['Pred_High'] = preds[:, 0]
    df_res['Pred_Low'] = preds[:, 1]
    
    # Mathematical correction: High must be >= Low
    df_res['Pred_High'] = np.maximum(df_res['Pred_High'], df_res['Pred_Low'])
    
    df_res.to_csv(output_file, index=False)
    print(f"Saved Interval Forecasts to {output_file}")

if __name__ == '__main__':
    main()
