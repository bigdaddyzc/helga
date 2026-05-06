import torch
from core.environment import EnvironmentEncoder

def test_environment_encoder_forward():
    """Test environment encoding produces correct shape."""
    encoder = EnvironmentEncoder()
    # Create mock input
    time_tensor = torch.randn(4)
    space_tensor = torch.randn(3)
    social_tensor = torch.randn(4)
    info_tensor = torch.randn(4)
    encoded = encoder(time_tensor, space_tensor, social_tensor, info_tensor)
    assert encoded.shape == (encoder.output_dim,)