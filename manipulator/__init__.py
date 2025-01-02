from gymnasium.envs.registration import register

register(
    id="TrunkManipulator-v0",
    entry_point="manipulator.trunk_environment:TrunkEnv",
    kwargs={
        "max_steps": 100,  # Add max_steps argument
        "target": [10.0, -30.0],  # Add the target argument
    }
)
