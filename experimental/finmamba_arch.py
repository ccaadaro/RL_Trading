import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Try to import optimized Mamba, fallback to simple LSTM/GRU if unavailable for compatibility
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

class InceptionBlock(nn.Module):
    """
    Paper Section 4.1: Market-Aware Graph Sparsification.
    Processes the Market Index M to determine sparsity level kappa.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.branch2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.fc = nn.Linear(out_channels * 3, 1)

    def forward(self, x):
        # x: (Batch, Channels, Length)
        b1 = F.relu(self.branch1(x))
        b2 = F.relu(self.branch2(x))
        b3 = F.relu(self.branch3(x))
        out = torch.cat([b1, b2, b3], dim=1)
        # Global Average Pooling over time dimension
        out = torch.mean(out, dim=2) 
        return torch.sigmoid(self.fc(out)) # Returns kappa (0 to 1)

class GraphAttentionLayer(nn.Module):
    """
    Paper Section 4.1: Graph Attention Aggregation (Equation 5 & 6)
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: (Batch, N, In_Features)
        # adj: (Batch, N, N) - Weighted adjacency matrix
        Wh = torch.matmul(h, self.W) # (B, N, Out)
        
        # Simple attention mechanism (optimized for batch)
        # Note: Full GAT attention usually involves N*N pairs. 
        # Here we use a simplified version compatible with weighted Adj from MAG.
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        
        # Mask using the sparsified adjacency matrix
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Aggregate
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        B, N, F_out = Wh.size()
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return self.leakyrelu(torch.matmul(all_combinations_matrix, self.a).view(B, N, N))

class MultiLevelMambaBlock(nn.Module):
    """
    Paper Section 4.2: Multi-Level Mamba.
    """
    def __init__(self, input_dim, d_model, n_levels=2):
        super().__init__()
        self.levels = n_levels
        self.projections = nn.ModuleList([
            nn.Linear(input_dim, d_model) for _ in range(n_levels)
        ])
        
        # Use Mamba if available, else fallback to LSTM
        if HAS_MAMBA:
            self.mamba_blocks = nn.ModuleList([
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) 
                for _ in range(n_levels)
            ])
        else:
            self.mamba_blocks = nn.ModuleList([
                nn.LSTM(d_model, d_model // 2, batch_first=True, bidirectional=True)
                for _ in range(n_levels)
            ])
            
        self.re_projections = nn.ModuleList([
            nn.Linear(d_model, input_dim) for _ in range(n_levels)
        ])

    def forward(self, x):
        # x: (Batch, N, Length, Features)
        B, N, L, F = x.shape
        # Flatten Batch and N for sequence processing: (B*N, L, F)
        x_flat = x.view(B * N, L, F)
        
        level_outputs = []
        for i in range(self.levels):
            # 1. Level Projection
            proj = self.projections[i](x_flat) # (B*N, L, d_model)
            
            # 2. Mamba/SSM Block
            if HAS_MAMBA:
                mamba_out = self.mamba_blocks[i](proj)
            else:
                mamba_out, _ = self.mamba_blocks[i](proj)
            
            # 3. Residual + Re-projection
            # Note: Paper Eq 8 implies combining projected and original
            # mamba_out is (B*N, L, d_model) if LSTM bidirectional=False or Mamba
            # If LSTM bidirectional=True, it is (B*N, L, d_model) because d_model//2 * 2
            
            out = self.re_projections[i](mamba_out + proj)
            level_outputs.append(out)
            
        # Concatenate levels
        concat = torch.cat(level_outputs, dim=-1) # (B*N, L, F * levels)
        
        # Reshape back
        return concat.view(B, N, L, -1)

class FinMamba(nn.Module):
    def __init__(self, num_nodes, feature_dim, market_dim, seq_len, 
                 d_model=64, n_levels=2, tau=10.0):
        super().__init__()
        self.tau = tau # Scaling factor for sparsity (Eq 3)
        self.num_nodes = num_nodes
        
        # --- 1. Market-Aware Graph ---
        self.inception = InceptionBlock(market_dim, 16)
        # GAT input is concat of Original (F) + Neighbor (F) -> 2*F? 
        # Paper says P = Stack(s || z). Here we map features before GAT for stability
        self.gat = GraphAttentionLayer(feature_dim, feature_dim)
        
        # --- 2. Multi-Level Mamba ---
        # Input to Mamba is Concat(Original, Graph_Embedding) -> 2 * feature_dim
        # The output of GAT is F, and we concat with original F -> 2*F
        self.mlm = MultiLevelMambaBlock(input_dim=2*feature_dim, d_model=d_model, n_levels=n_levels)
        
        # Final prediction
        # Output of MLM is (B, N, L, n_levels * 2 * feature_dim)
        # Because re_projections maps back to input_dim (2*feature_dim)
        self.regressor = nn.Sequential(
            nn.Linear(n_levels * 2 * feature_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1) # Ranking score y^t
        )

    def forward(self, x, market_index, prior_adj, posterior_adj):
        """
        x: (Batch, N, L, F) - Stock Features
        market_index: (Batch, Market_Dim, L)
        prior_adj: (N, N) - Industry Decay Matrix D (Static)
        posterior_adj: (Batch, N, N) - Correlation Matrix Q (Dynamic)
        """
        B, N, L, F = x.shape
        
        # --- A. Market-Aware Graph Sparsification ---
        # 1. Calculate Sparsity Level kappa (Eq 3)
        kappa = self.inception(market_index) * self.tau # (Batch, 1)
        
        # 2. Combine Adjacency (Eq 2)
        # A = Q * D
        adj = posterior_adj * prior_adj.unsqueeze(0) # (Batch, N, N)
        
        # 3. Pruning (Eq 4)
        # Keep Top K edges based on kappa
        # We create a differentiable soft mask or hard mask. 
        # For training, hard mask with gradient straight-through or just simple masking.
        # Implementation: zero out elements below k-th percentile value
        
        # Convert kappa (0-1) to integer K
        k_val = (kappa * (N * N)).long().clamp(min=N, max=N*N) # Ensure at least self-loops/minimal connectivity
        
        # Create mask (inefficient for large N, fine for <500 stocks)
        mask = torch.zeros_like(adj)
        for b in range(B):
            flattened = adj[b].view(-1)
            k = k_val[b].item()
            if k < 1: k = 1
            topk_vals, _ = torch.topk(flattened, k)
            threshold = topk_vals[-1]
            mask[b] = (adj[b] >= threshold).float()
            
        final_adj = adj * mask
        
        # --- B. Graph Attention Aggregation ---
        # GAT operates on the last time step features usually, or we apply it per step.
        # Paper implies applying it to generate representations. 
        # For efficiency, we apply GAT on the features of the last time step L.
        # Alternatively, apply GAT to the whole sequence feature vector.
        
        # Let's apply GAT on the most recent features x[:, :, -1, :]
        curr_feat = x[:, :, -1, :] # (B, N, F)
        neighbor_feat = self.gat(curr_feat, final_adj) # (B, N, F)
        
        # Broadcast neighbor features across time L for concatenation
        neighbor_feat_expanded = neighbor_feat.unsqueeze(2).repeat(1, 1, L, 1) # (B, N, L, F)
        
        # Stack stock embedding P (Eq 6/End of 4.1)
        p = torch.cat([x, neighbor_feat_expanded], dim=-1) # (B, N, L, 2F)
        
        # --- C. Multi-Level Mamba ---
        # Process sequence dependencies
        mamba_out = self.mlm(p) # (B, N, L, Out_Dim)
        
        # Take the last time step output for prediction
        final_embed = mamba_out[:, :, -1, :]
        
        # Score Prediction
        scores = self.regressor(final_embed) # (B, N, 1)
        
        # Return scores and neighbor_feat (for GIB loss)
        # neighbor_feat is (B, N, F), which matches original_input dimensions
        return scores.squeeze(-1), neighbor_feat