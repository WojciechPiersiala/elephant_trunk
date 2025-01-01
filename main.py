#!/home/wp/Studia/soft_robotics/elephant_trunk/trunk/bin/python
import matplotlib.pyplot as plt
from manipulator.manipulator import Manipulator
from manipulator.trunk_environment import TrunkEnv
import gymnasium as gym
from pynput.keyboard import Key, Listener
import random
import time

def main():
    rgb_array = False
    if rgb_array:
        env = gym.make("TrunkManipulator-v0", render_mode="rgb_array", max_steps=50)
        env = gym.wrappers.RecordVideo(env, video_folder="trunk-episodes", name_prefix="eval", episode_trigger=lambda x: x == 0)
    else:
        env = gym.make("TrunkManipulator-v0", render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env=env)

    try:
        state = env.reset()
        done = False
        start_time = time.time()
        while not done:
            action = env.action_space.sample()  # Random action
            state, reward, terminate, truncated, info = env.step(action)
            done = terminate or truncated
            env.render()
        end_time = time.time() 
    finally:
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds, rgb_array: {rgb_array}")
        # Ensure the environment is properly closed
        env.close()
if __name__ == "__main__":
    main()
