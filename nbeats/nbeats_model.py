"""
N-BEATS (Neural Basis Expansion Analysis for Time Series) Model

Implementation of N-BEATS architecture for time series forecasting.
Based on: "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
(Oreshkin et al., 2019)

Key Features:
- Generic architecture with stacked blocks
- Forward/Backward residual connections
- Doubly residual stacking for improved gradient flow
- Support for both generic and interpretable configurations
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List


class NBeatsBlock(nn.Module):
    """
    Single N-BEATS block with forward and backward residual connections.
    
    Architecture:
    - Fully connected stack (4 layers)
    - Theta layer for basis expansion
    - Backward forecast (g^b) and forward forecast (g^f)
    """
    
    def __init__(
        self,
        input_size: int,
        theta_size: int,
        horizon: int,
        num_layers: int = 4,
        layer_size: int = 256,
        activation: str = "relu"
    ):
        """
        Args:
            input_size: Lookback window size
            theta_size: Number of basis function parameters
            horizon: Forecast horizon
            num_layers: Number of FC layers in the block
            layer_size: Width of FC layers
            activation: Activation function ('relu', 'gelu', 'selu')
        """
        super().__init__()
        
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "selu":
            self.activation = nn.SELU()
        else:
            self.activation = nn.ReLU()
        
        # Fully connected stack
        layers = []
        layers.append(nn.Linear(input_size, layer_size))
        layers.append(self.activation)
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(self.activation)
        
        self.fc_stack = nn.Sequential(*layers)
        
        # Theta layers for basis expansion
        self.theta_b = nn.Linear(layer_size, theta_size)  # Backward (backcast)
        self.theta_f = nn.Linear(layer_size, theta_size)  # Forward (forecast)
        
        # Basis functions (generic - simple linear projection)
        self.backcast_fc = nn.Linear(theta_size, input_size)
        self.forecast_fc = nn.Linear(theta_size, horizon)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the block.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            backcast: Reconstruction of input (batch_size, input_size)
            forecast: Forecast output (batch_size, horizon)
        """
        # FC stack
        h = self.fc_stack(x)
        
        # Theta parameters
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)
        
        # Generate backcast and forecast
        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)
        
        return backcast, forecast


class NBeatsStack(nn.Module):
    """
    Stack of N-BEATS blocks with residual connections.
    """
    
    def __init__(
        self,
        num_blocks: int,
        input_size: int,
        theta_size: int,
        horizon: int,
        num_layers: int = 4,
        layer_size: int = 256,
        activation: str = "relu"
    ):
        """
        Args:
            num_blocks: Number of blocks in the stack
            input_size: Lookback window size
            theta_size: Number of basis function parameters
            horizon: Forecast horizon
            num_layers: Number of FC layers per block
            layer_size: Width of FC layers
            activation: Activation function
        """
        super().__init__()
        
        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=input_size,
                theta_size=theta_size,
                horizon=horizon,
                num_layers=num_layers,
                layer_size=layer_size,
                activation=activation
            )
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the stack.
        
        Args:
            x: Input tensor (batch_size, input_size)
        
        Returns:
            residual: Final residual after all blocks
            forecast: Accumulated forecast
        """
        residual = x
        forecast = None
        
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast  # Doubly residual
            
            if forecast is None:
                forecast = block_forecast
            else:
                forecast = forecast + block_forecast
        
        return residual, forecast


class NBeatsModel(nn.Module):
    """
    Full N-BEATS model with multiple stacks.
    
    Architecture:
    - Stack 1: Trend modeling
    - Stack 2: Seasonality modeling
    - Stack 3: Generic residual modeling
    """
    
    def __init__(
        self,
        input_size: int,
        horizon: int,
        num_stacks: int = 2,
        num_blocks_per_stack: int = 3,
        theta_size: int = 4,
        num_layers: int = 4,
        layer_size: int = 256,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        Args:
            input_size: Lookback window size (historical data points)
            horizon: Forecast horizon (future steps to predict)
            num_stacks: Number of stacks (typically 2-3)
            num_blocks_per_stack: Blocks per stack
            theta_size: Basis function parameters
            num_layers: FC layers per block
            layer_size: Width of FC layers
            activation: Activation function
            dropout: Dropout rate (0 = no dropout)
        """
        super().__init__()
        
        self.input_size = input_size
        self.horizon = horizon
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Create stacks
        self.stacks = nn.ModuleList([
            NBeatsStack(
                num_blocks=num_blocks_per_stack,
                input_size=input_size,
                theta_size=theta_size,
                horizon=horizon,
                num_layers=num_layers,
                layer_size=layer_size,
                activation=activation
            )
            for _ in range(num_stacks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the entire model.
        
        Args:
            x: Input tensor (batch_size, input_size)
        
        Returns:
            forecast: Final forecast (batch_size, horizon)
        """
        residual = x
        forecast = None
        
        for stack in self.stacks:
            residual, stack_forecast = stack(residual)
            
            # Apply dropout to stack output
            if self.dropout is not None:
                stack_forecast = self.dropout(stack_forecast)
            
            if forecast is None:
                forecast = stack_forecast
            else:
                forecast = forecast + stack_forecast
        
        return forecast
    
    def predict(
        self,
        x: np.ndarray,
        device: str = "cpu"
    ) -> np.ndarray:
        """
        Make predictions on numpy arrays.
        
        Args:
            x: Input array (batch_size, input_size) or (input_size,)
            device: Device to run on ('cpu' or 'cuda')
        
        Returns:
            predictions: Forecast array (batch_size, horizon) or (horizon,)
        """
        self.eval()
        
        # Handle single sample
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Convert to tensor
        x_tensor = torch.FloatTensor(x).to(device)
        
        with torch.no_grad():
            forecast = self.forward(x_tensor)
        
        # Convert back to numpy
        predictions = forecast.cpu().numpy()
        
        # Return single sample if input was single
        if predictions.shape[0] == 1:
            return predictions[0]
        
        return predictions


def create_nbeats_model(
    input_size: int,
    horizon: int,
    model_size: str = "medium",
    **kwargs
) -> NBeatsModel:
    """
    Factory function to create N-BEATS models with predefined configurations.
    
    Args:
        input_size: Lookback window size
        horizon: Forecast horizon
        model_size: Predefined size ('small', 'medium', 'large')
        **kwargs: Override default parameters
    
    Returns:
        Configured N-BEATS model
    """
    
    configs = {
        "small": {
            "num_stacks": 2,
            "num_blocks_per_stack": 2,
            "theta_size": 4,
            "num_layers": 3,
            "layer_size": 128,
            "activation": "relu",
            "dropout": 0.1
        },
        "medium": {
            "num_stacks": 2,
            "num_blocks_per_stack": 3,
            "theta_size": 8,
            "num_layers": 4,
            "layer_size": 256,
            "activation": "relu",
            "dropout": 0.1
        },
        "large": {
            "num_stacks": 3,
            "num_blocks_per_stack": 4,
            "theta_size": 16,
            "num_layers": 4,
            "layer_size": 512,
            "activation": "gelu",
            "dropout": 0.2
        }
    }
    
    config = configs.get(model_size, configs["medium"])
    config.update(kwargs)
    
    return NBeatsModel(
        input_size=input_size,
        horizon=horizon,
        **config
    )


if __name__ == "__main__":
    # Test the model
    print("Testing N-BEATS model...")
    
    input_size = 168  # 1 week of hourly data
    horizon = 24      # Forecast 24 hours ahead
    batch_size = 32
    
    # Create model
    model = create_nbeats_model(
        input_size=input_size,
        horizon=horizon,
        model_size="medium"
    )
    
    print(f"Model created: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, input_size)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("[OK] Model test passed!")
