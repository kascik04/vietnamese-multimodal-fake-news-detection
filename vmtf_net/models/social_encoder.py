"""
Social Metrics Encoder module for VMTF-Net.
Processes numerical social interaction engagement features 
(share count, like count, comment count, reaction ratios) via MLP.
"""

import torch
import torch.nn as nn

from ..config import VMTFConfig


class SocialMetricsEncoder(nn.Module):
    """
    Multilayer Perceptron (MLP) mapping normalized social engagement 
    features to a dense representation space E_S in R^128.
    
    Structure:
        Input [B, D_in] -> Linear(64) -> BatchNorm1d -> ReLU -> Dropout
                        -> Linear(128) -> BatchNorm1d -> ReLU -> Dropout -> E_S [B, 128]
    """

    def __init__(self, config: VMTFConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.social_input_dim
        out_dim = config.social_hidden_dim
        dropout = config.social_dropout
        intermediate_dim = 64

        self.net = nn.Sequential(
            nn.Linear(in_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, social_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for social metrics.
        
        Args:
            social_features: Normalized numerical engagement tensor [batch_size, social_input_dim]
            
        Returns:
            Dense social embedding tensor E_S of shape [batch_size, social_hidden_dim (128)]
        """
        # Handle single sample during train mode if batch size is 1 to avoid BatchNorm error
        if self.training and social_features.size(0) == 1:
            self.eval()
            out = self.net(social_features)
            self.train()
            return out

        return self.net(social_features)
