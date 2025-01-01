from gymnasium.envs.registration import register

register(
    id="TrunkManipulator-v0",
    entry_point="manipulator.trunk_environment:TrunkEnv",
)
# print(f"registerd as :TrunkManipulator")


# register(
#     id="gymnasium_env/GridWorld-v0",
#     entry_point="gymnasium_env.envs:GridWorldEnv",
# )