"""Environment variable encoder for nested time/space/social/info dimensions."""

import torch
import torch.nn as nn
from typing import Tuple


class EnvironmentEncoder(nn.Module):
    """Encode nested environment variables into a unified vector.

    Architecture:
        E = [ Enc(E_time); Enc(E_space); Enc(E_social); Enc(E_info) ] -> R^d

    Attributes:
        time_dim: Input dim for time variables (4)
        space_dim: Input dim for space variables (3)
        social_dim: Input dim for social variables (4)
        info_dim: Input dim for info variables (4)
        hidden_dim: Hidden layer dim
        output_dim: Final encoded vector dim
    """

    def __init__(
        self,
        time_dim: int = 4,
        space_dim: int = 3,
        social_dim: int = 4,
        info_dim: int = 4,
        hidden_dim: int = 32,
        output_dim: int = 32
    ):
        super().__init__()
        self.time_dim = time_dim
        self.space_dim = space_dim
        self.social_dim = social_dim
        self.info_dim = info_dim
        self.output_dim = output_dim

        # Separate encoders per dimension
        self.time_encoder = nn.Linear(time_dim, hidden_dim)
        self.space_encoder = nn.Linear(space_dim, hidden_dim)
        self.social_encoder = nn.Linear(social_dim, hidden_dim)
        self.info_encoder = nn.Linear(info_dim, hidden_dim)

        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 4, output_dim)

    def forward(
        self,
        time: torch.Tensor,
        space: torch.Tensor,
        social: torch.Tensor,
        info: torch.Tensor
    ) -> torch.Tensor:
        """Encode environment variables.

        Args:
            time: Time variables (batch, 4)
            space: Space variables (batch, 3)
            social: Social variables (batch, 4)
            info: Info variables (batch, 4)

        Returns:
            Encoded environment vector (batch, output_dim)
        """
        time_enc = torch.relu(self.time_encoder(time))
        space_enc = torch.relu(self.space_encoder(space))
        social_enc = torch.relu(self.social_encoder(social))
        info_enc = torch.relu(self.info_encoder(info))

        # Concatenate and fuse
        combined = torch.cat([time_enc, space_enc, social_enc, info_enc], dim=-1)
        output = self.fusion(combined)
        return output