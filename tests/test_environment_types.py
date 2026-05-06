import sys
sys.path.insert(0, '..')

import numpy as np
from core.types import EnvironmentVariables, TimeVariables, SpaceVariables, SocialVariables, InfoVariables

def test_environment_variables_creation():
    """Test nested environment variables structure."""
    env = EnvironmentVariables(
        time=TimeVariables(current_time=0.5, deadline_pressure=0.3, time_horizon=0.7, duration=0.4),
        space=SpaceVariables(location_type=0.2, physical_context=0.1, proximity=0.3),
        social=SocialVariables(relationship_dynamics=0.5, power_distance=0.6, group_norm=0.4, cultural_context=0.3),
        info=InfoVariables(uncertainty=0.5, confidence=0.7, information_quality=0.6, missing_key_info=False)
    )
    assert env.time.current_time == 0.5
    assert env.info.missing_key_info == False