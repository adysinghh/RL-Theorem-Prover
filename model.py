import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple


class PolicyNet(nn.Module):
    """
    A simple MLP policy network for discrete action selection.

    Attributes:
        fc1: First linear layer.
        fc2: Second linear layer producing action logits.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        """
        Initialize the policy network.

        Args:
            state_dim: Dimension of the input state vector.
            action_dim: Number of discrete actions.
            hidden_dim: Number of hidden units.
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute action logits.

        Args:
            x: Input tensor of shape (batch_size, state_dim).
        Returns:
            logits: Tensor of shape (batch_size, action_dim).
        """
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Sample an action from the policy network given a single state.

        Args:
            state: 1D tensor of shape (state_dim,).
        Returns:
            action: Selected action index.
            log_prob: Log probability of the selected action.
        """
        logits = self.forward(state.unsqueeze(0))  # add batch dim
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).squeeze(0)

    def save(self, path: str) -> None:
        """
        Save the model parameters to disk.

        Args:
            path: File path for saving state dict.
        """
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path: str, state_dim: int, action_dim: int, hidden_dim: int = 128) -> "PolicyNet":
        """
        Load model parameters from disk and return a PolicyNet instance.

        Args:
            path: File path to the saved state dict.
            state_dim: Dimension of the input state vector.
            action_dim: Number of discrete actions.
            hidden_dim: Number of hidden units.
        Returns:
            model: PolicyNet with loaded weights.
        """
        model = PolicyNet(state_dim, action_dim, hidden_dim)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
