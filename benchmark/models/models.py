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



import torch
import torch.nn as nn

__author__ = "Samuel Adebayo"

class SharedEncoder(nn.Module):
    """
    Temporal backbone with interchangeable flavour.
    """
    def __init__(self, arch: str='biLSTM', d_model: int=128, num_layers: int=2):
        super().__init__()
        self.arch = arch

        if arch.lower() == 'bilstm':
            self.core = nn.LSTM(input_size=2*42, hidden_size=d_model, num_layers=num_layers, batch_first=True, bidirectional=True)
            self.out_dim = 2*d_model

        elif arch.lower() == 'bigru':
            self.core = nn.GRU(input_size=2*42, hidden_size=d_model, num_layers=num_layers, batch_first=True, bidirectional=True)
            self.out_dim = 2*d_model

        elif arch.lower() == 'tcn':
            layers, channels = [], []
            for dil in (1, 2, 4, 8):
                layers += [nn.Conv1d(2*42 if dil == 1 else channels, d_model,
                                     kernel_size=3, dilation=dil, padding=dil),
                           nn.ReLU(),
                           nn.LayerNorm([d_model, 60])]
                self.core = nn.Sequential(*layers)
                self.out_dim = d_model

        elif arch.lower() == 'transformer':
            enc_layer = nn.TransformerEncoderLayer(d_model, nhead=8,
                                                   dim_feedforward=512, activation='relu',
                                                   batch_first=True)
            self.pos = nn.Parameter(torch.randn(1, 60, d_model))
            self.core = nn.TransformerEncoder(enc_layer, num_layers=6)
            self.fc_in = nn.Linear(2*42, d_model)
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
        else:
            x = self.core(self.fc_in(x) + self.pos) # (batch, seq_len, d_model)
            feat = x
        return feat

class TwoHeadNet(nn.Module):
    """
    Shared encode + pose / intent heads.
    """
    def __init__(self, arch='bilstm'):
        super().__init__()
        self.encoder = SharedEncoder(arch=arch)
        f = self.encoder.out_dim
        self.pose_head = nn.Sequential(nn.Linear(f, 512), nn.ReLU(),
                                       nn.Linear(512, 84))
        self.intent_head = nn.Sequential(nn.Linear(f, 128), nn.ReLU(),
                                         nn.Linear(128, 136))

    def forward(self, x):
        y = self.encoder(x)
        pred_seq = self.pose_head(y)
        logits = self.intent_head(y[:, -1])
        return pred_seq, logits

