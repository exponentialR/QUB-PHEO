"""
Baseline backbones for
    *Establishing Baselines for Dyadic Visual Motion Prediction
     Using the QUB-PHEO Dataset*  (IFAC J3C 2025)

Author: Samuel Adebayo  <samuel.adebayo@qub.ac.uk>
Last updated: 17 May 2025

The file exposes two classes:

1. SharedEncoder
   A temporal backbone whose flavour is selected by the *arch*
   argument:
       - "bilstm"        - 2-layer bidirectional LSTM
       - "bigru"         - 2-layer bidirectional GRU
       - "tcn"           - 4-block dilated Temporal CNN
       - "transformer"   - 6-layer Transformer encoder (8 heads)

   All variants accept a tensor of shape
       (batch, 60 frames, 84 features)
   produced by the QUB-PHEO aerial-view loader and return a sequence
   of hidden features with dimension *out_dim* (128 or 256).

2. TwoHeadNet
   Wraps a SharedEncoder and adds:
       • pose_head     → predicts the 60-frame hand-landmark trajectory
       • intent_head   → predicts the discrete sub-task ID

The network is used in Section 4 (Baseline Architectures) of the paper
and trained with the loss in Eq. (5).  See train.py for usage.
"""

__author__ = "Samuel Adebayo"

import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, Kt=9):
        super().__init__()
        # A: (V, V) numpy or torch adjacency matrix for 42 joints
        self.register_buffer('A', A)
        # 1×1 “spatial” conv is just a linear mixing of node features
        self.conv_spatial = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # temporal conv: (Kt × 1) over time dimension
        self.conv_temporal = nn.Conv2d(out_channels, out_channels,
                                       kernel_size=(Kt, 1),
                                       padding=(Kt//2, 0))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = x.permute(0,2,3,1)              # (B, T, V, C)
        x = torch.matmul(self.A, x)        # graph mix: (B, T, V, C)
        x = x.permute(0,3,1,2)              # (B, C, T, V)
        x = F.relu(self.conv_spatial(x))
        #  2) temporal conv
        x = self.conv_temporal(x)           # (B, C, T, V)
        return F.relu(self.bn(x))


class STGCNEncoder(nn.Module):
    def __init__(self, A, in_channels=2, hidden_channels=[64,128,256]):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in hidden_channels:
            layers.append(STGCNBlock(c_in, c_out, A))
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_channels[-1] * 42  # will flatten joints into feature dim

    def forward(self, x):
        # x: (B, T, 84)  where 84 = 42 joints × 2 coords
        B,T,D = x.shape
        V = D // 2
        # reshape to (B, C=2, T, V)
        x = x.view(B, T, V, 2).permute(0,3,1,2)
        # apply ST-GCN stack
        y = self.net(x)                   # (B, C_last, T, V)
        # flatten back to (B, T, V*C_last)
        y = y.permute(0,2,3,1).contiguous()
        return y.view(B, T, -1)


class SharedEncoder(nn.Module):
    """
    Temporal backbone with interchangeable flavour.
    """
    def __init__(self, arch: str='biLSTM', input_dim: int = 84, d_model: int=128, num_layers: int=2, dropout: float=0.1, A=None):
        super().__init__()
        self.arch = arch
        self.dropout = dropout

        if arch.lower() == 'bilstm':
            self.core = nn.LSTM(input_size=input_dim, hidden_size=d_model, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=self.dropout if num_layers > 1 else 0)
            self.out_dim = 2*d_model

        elif arch.lower() == 'bigru':
            self.core = nn.GRU(input_size=input_dim, hidden_size=d_model, num_layers=num_layers, batch_first=True, bidirectional=True)
            self.out_dim = 2*d_model

        elif arch.lower() == 'stgcn':
            assert A is not None, "Must pass adjacency A for STGCN"
            self.core = STGCNEncoder(A, in_channels=2, hidden_channels=[64, 128, 256])
            self.out_dim = self.core.out_dim


        elif arch.lower() == 'tcn':
            layers = []
            in_channels = input_dim # Default=> 84
            for dilation in (1, 2, 4, 8):
                layers += [
                    nn.Conv1d(in_channels, d_model,
                              kernel_size=3,
                              dilation=dilation,
                              padding=dilation),
                              nn.ReLU(),
                              nn.LayerNorm([d_model, 60])
                ]
                in_channels = d_model
            self.core = nn.Sequential(*layers)
            self.out_dim = d_model

        elif arch.lower() == 'transformer':
            enc_layer = nn.TransformerEncoderLayer(d_model, nhead=8,
                                                   dim_feedforward=512, activation='relu',
                                                   batch_first=True)
            self.pos = nn.Parameter(torch.randn(1, 60, d_model))
            self.core = nn.TransformerEncoder(enc_layer, num_layers=6)
            self.fc_in = nn.Linear(input_dim, d_model)
            self.out_dim = d_model
        else:
            raise ValueError(f"Unknown architecture: {arch}. Supported architectures are: 'biLSTM', 'biGRU', 'TCN', 'Transformer'.")

    def forward(self, x):
        if self.arch in {'bilstm', 'bigru'}:
            x, _ = self.core(x) # (batch, seq_len, 2*42)
            feat = x
        elif self.arch == 'tcn':
            x = self.core(x.transpose(1, 2)) # Conv1D expects (batch, channels, seq_len)
            feat = x.transpose(1, 2)
        elif self.arch  == 'stgcn':
            feat = self.core(x)
        else:
            x = self.core(self.fc_in(x) + self.pos) # (batch, seq_len, d_model)
            feat = x
        return feat

class TwoHeadNet(nn.Module):
    """
    Shared encode + pose / intent heads.
    """
    def __init__(self, arch, input_dim, dropout=0.1, A=None):
        super().__init__()

        self.encoder = SharedEncoder(arch=arch, input_dim=input_dim, dropout=dropout, A=A)
        f = self.encoder.out_dim
        self.pose_head = nn.Sequential(nn.LayerNorm(f),
                                       nn.Dropout(dropout),
                                       nn.Linear(f, 84))
        self.intent_head = nn.Sequential(nn.LayerNorm(f),
                                        nn.Dropout(self.encoder.dropout),
                                         nn.Linear(f, f//2),
                                         nn.LeakyReLU(),
                                         nn.Linear(f//2, 36))

    def forward(self, x, use_intent_head=False):
        y = self.encoder(x)
        pred_seq = self.pose_head(y)
        if use_intent_head:
            pooled = y.mean(dim=1)
            logits = self.intent_head(pooled)
            return pred_seq, logits
        else:
            return pred_seq

