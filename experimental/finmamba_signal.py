import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from finmamba_arch import FinMamba

# Configuration for the internal supervised training loop
TRAIN_CONFIG = {
    "epochs": 50,           # Increased to ensure convergence
    "batch_size": 64,      # Batch size
    "lookback": 60,        # L
    "lr": 0.0001,          # Lower LR for stability
    "lambda_gib": 0.1,     # Weight for GIB loss
    "eta_rank": 0.5,       # Weight for Rank loss vs MSE
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def construct_industry_decay_matrix(universe, sector_map):
    """
    Paper Eq 2: Prior Matrix D based on Primary/Secondary industries.
    """
    n = len(universe)
    D = np.ones((n, n), dtype=np.float32)
    
    # If universe is empty or N=1, return 1x1 matrix
    if n <= 1:
        return torch.tensor(D)

    delta1 = 0.8
    delta2 = 0.5
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            s1 = universe[i]
            s2 = universe[j]
            
            sec1 = sector_map.get(s1, {}).get('secondary', 'A')
            sec2 = sector_map.get(s2, {}).get('secondary', 'B')
            
            prim1 = sector_map.get(s1, {}).get('primary', 'X')
            prim2 = sector_map.get(s2, {}).get('primary', 'Y')
            
            if sec1 == sec2:
                D[i, j] = 1.0
            elif prim1 == prim2:
                D[i, j] = delta1
            else:
                D[i, j] = delta2
    return torch.tensor(D)

def loss_function(pred_scores, true_returns, embeddings, original_input, eta, lam):
    """
    Paper Section 4.3: Eq 9 (Rank Loss) and Eq 10 (GIB Loss).
    """
    # 1. Rank & Prediction Loss (Eq 9)
    # MSE Part
    mse = torch.mean((pred_scores - true_returns) ** 2)
    
    # Pairwise Ranking Part
    # Efficient broadcasting for pairwise differences
    # If N=1, this part is 0 because pred_diff and true_diff are 0
    if pred_scores.shape[1] > 1:
        pred_diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(2) # (B, N, N)
        true_diff = true_returns.unsqueeze(1) - true_returns.unsqueeze(2)
        
        # max(0, - (y_i - y_j) * (r_i - r_j))
        rank_loss = torch.relu(-(pred_diff * true_diff))
        rank_loss = torch.mean(rank_loss)
    else:
        rank_loss = torch.tensor(0.0, device=pred_scores.device)
    
    L_rp = mse + eta * rank_loss
    
    # 2. Graph Information Bottleneck (GIB) Loss (Eq 10)
    # z: embeddings (B, N, Features)
    # s: original input (last step) (B, N, Features)
    
    z_mean = embeddings.mean(dim=0) # (N, F)
    s_mean = original_input.mean(dim=0)
    
    z_var = embeddings.var(dim=0) + 1e-6
    s_var = original_input.var(dim=0) + 1e-6
    
    numerator = torch.norm(z_mean - s_mean, dim=1) ** 2
    denominator = z_var.mean(dim=1) + s_var.mean(dim=1) 
    
    L_gib = torch.mean(numerator / denominator)
    
    return L_rp + lam * L_gib

def create_sequences(data, market_data, target, lookback):
    """
    Create sequences for training/inference.
    data: (T, F)
    market_data: (T, M)
    target: (T,)
    Returns: (B, L, F), (B, L, M), (B,)
    """
    xs, ms, ys = [], [], []
    # Ensure we have enough data
    if len(data) <= lookback:
        return torch.empty(0, lookback, data.shape[1]), torch.empty(0, lookback, market_data.shape[1]), torch.empty(0)

    for i in range(len(data) - lookback):
        xs.append(data[i:i+lookback])
        ms.append(market_data[i:i+lookback])
        ys.append(target[i+lookback-1]) # Target at the end of the window
    
    return torch.stack(xs), torch.stack(ms), torch.stack(ys)

def train_finmamba(model, train_loader, optimizer, device, prior_adj):
    model.train()
    total_loss = 0
    count = 0
    for x_batch, m_batch, y_batch in train_loader:
        # x_batch: (B, L, F) -> Need (B, N, L, F)
        # m_batch: (B, L, M) -> Need (B, M, L) for Inception
        
        B, L, F = x_batch.shape
        N = 1 # Single pair
        
        x_in = x_batch.unsqueeze(1).to(device) # (B, 1, L, F)
        m_in = m_batch.permute(0, 2, 1).to(device) # (B, M, L)
        y_true = y_batch.unsqueeze(1).to(device) # (B, 1)
        
        # Posterior Adj (Q) - Dynamic correlation
        # For N=1, it's just [[1.0]]
        posterior_adj = torch.ones((B, N, N), device=device)
        
        optimizer.zero_grad()
        
        scores, embeddings = model(x_in, m_in, prior_adj.to(device), posterior_adj)
        
        # Original input at last step for GIB loss
        original_input = x_in[:, :, -1, :] # (B, N, F)
        
        loss = loss_function(scores, y_true, embeddings, original_input, TRAIN_CONFIG['eta_rank'], TRAIN_CONFIG['lambda_gib'])
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
        
    return total_loss / count if count > 0 else 0

def get_finmamba_signal_features(df, config):
    """
    Main entry point called by train_rl.py.
    """
    print(f"🐍 Initializing FinMamba Feature Extraction...")
    
    device = TRAIN_CONFIG['device']
    lookback = TRAIN_CONFIG['lookback']
    
    # 1. Prepare Data
    # Select features
    # Avoid leakage?
    # Let's use all numeric columns that are likely features.
    feature_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.float32] and c not in ['open', 'high', 'low', 'close', 'volume']]
    if not feature_cols:
        feature_cols = ['close', 'volume'] # Fallback
        
    market_cols = ['vix_index_feature', 'dxy_feature']
    market_cols = [c for c in market_cols if c in df.columns]
    if not market_cols:
        market_cols = ['close'] # Fallback
        
    target_col = config.get("finmamba_features", {}).get("base_return_column", "close_return_feature")
    if target_col not in df.columns:
        print(f"⚠️ Target column {target_col} not found. Using dummy target.")
        df[target_col] = 0.0
        
    # Scale
    scaler = RobustScaler()
    market_scaler = RobustScaler()

    # Fix leakage: Fit on train split only
    train_ratio = config.get("data_params", {}).get("train_split", 0.7)
    split_idx = int(len(df) * train_ratio)
    print(f"⚖️ Fitting scalers on first {split_idx} rows (Train Split) to avoid leakage.")

    scaler.fit(df[feature_cols].iloc[:split_idx].fillna(0))
    data_values = scaler.transform(df[feature_cols].fillna(0))

    market_scaler.fit(df[market_cols].iloc[:split_idx].fillna(0))
    market_values = market_scaler.transform(df[market_cols].fillna(0))

    target_values = df[target_col].shift(-1).fillna(0).values # Predict next return
    
    data_tensor = torch.tensor(data_values, dtype=torch.float32)
    market_tensor = torch.tensor(market_values, dtype=torch.float32)
    target_tensor = torch.tensor(target_values, dtype=torch.float32)
    
    # 2. Initialize Model
    N = 1 # Single pair
    F_dim = len(feature_cols)
    M_dim = len(market_cols)
    
    model = FinMamba(num_nodes=N, feature_dim=F_dim, market_dim=M_dim, seq_len=lookback).to(device)
    
    cache_path = Path(__file__).parent / "checkpoints" / "finmamba_model.pt"
    os.makedirs(cache_path.parent, exist_ok=True)
    
    prior_adj = torch.ones((N, N), device=device) # Identity for N=1
    
    # 3. Train or Load
    if cache_path.exists():
        try:
            state_dict = torch.load(cache_path, map_location=device)
            # Check if dimensions match
            # state_dict keys: inception..., gat.W, mlm...
            if 'gat.W' in state_dict and state_dict['gat.W'].shape[0] == F_dim: 
                model.load_state_dict(state_dict)
                model.eval()
                print("✅ Loaded Pretrained FinMamba Model.")
            else:
                print("⚠️ Dimension mismatch in checkpoint. Retraining...")
                # Force retraining
                raise ValueError("Dimension mismatch")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}. Retraining...")
            # Train
            train_split = int(len(data_tensor) * config.get("data_params", {}).get("train_split", 0.7))
            train_x = data_tensor[:train_split]
            train_m = market_tensor[:train_split]
            train_y = target_tensor[:train_split]
            
            x_seq, m_seq, y_seq = create_sequences(train_x, train_m, train_y, lookback)
            if len(x_seq) > 0:
                dataset = TensorDataset(x_seq, m_seq, y_seq)
                loader = DataLoader(dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
                
                optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])
                
                print(f"🚀 Training FinMamba for {TRAIN_CONFIG['epochs']} epochs...")
                for epoch in range(TRAIN_CONFIG['epochs']):
                    loss = train_finmamba(model, loader, optimizer, device, prior_adj)
                    print(f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']} - Loss: {loss:.4f}")
                    
                torch.save(model.state_dict(), cache_path)
                print("💾 Saved FinMamba Model.")
            else:
                print("⚠️ Not enough data to train.")
            
    else:
        print("⚠️ No checkpoint found. Training...")
        train_split = int(len(data_tensor) * config.get("data_params", {}).get("train_split", 0.7))
        train_x = data_tensor[:train_split]
        train_m = market_tensor[:train_split]
        train_y = target_tensor[:train_split]
        
        x_seq, m_seq, y_seq = create_sequences(train_x, train_m, train_y, lookback)
        if len(x_seq) > 0:
            dataset = TensorDataset(x_seq, m_seq, y_seq)
            loader = DataLoader(dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
            
            optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])
            
            print(f"🚀 Training FinMamba for {TRAIN_CONFIG['epochs']} epochs...")
            for epoch in range(TRAIN_CONFIG['epochs']):
                loss = train_finmamba(model, loader, optimizer, device, prior_adj)
                print(f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']} - Loss: {loss:.4f}")
            
            torch.save(model.state_dict(), cache_path)
            print("💾 Saved FinMamba Model.")
        else:
            print("⚠️ Not enough data to train.")

    # 4. Inference
    model.eval()
    print("🔮 Running FinMamba Inference...")
    
    # Process in batches to avoid OOM
    all_scores = []
    
    x_seq_all, m_seq_all, _ = create_sequences(data_tensor, market_tensor, target_tensor, lookback)
    
    if len(x_seq_all) == 0:
        return pd.DataFrame(index=df.index, data={"finmamba_score_feature": 0.0})

    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(x_seq_all), batch_size):
            x_b = x_seq_all[i:i+batch_size].unsqueeze(1).to(device) # (B, 1, L, F)
            m_b = m_seq_all[i:i+batch_size].permute(0, 2, 1).to(device) # (B, M, L)
            
            posterior_adj = torch.ones((x_b.shape[0], N, N), device=device)
            
            scores, _ = model(x_b, m_b, prior_adj.to(device), posterior_adj)
            all_scores.append(scores.cpu().numpy())
            
    scores_flat = np.concatenate(all_scores).flatten()
    
    # Pad the beginning (first lookback elements are missing)
    pad_len = len(df) - len(scores_flat)
    full_scores = np.concatenate([np.zeros(pad_len), scores_flat])
    
    output_df = pd.DataFrame(index=df.index)
    output_df["finmamba_score_feature"] = full_scores
    
    return output_df
