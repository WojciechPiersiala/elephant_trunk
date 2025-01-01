#!/home/wp/Studia/soft_robotics/elephant_trunk/trunk/bin/python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = spaces.Box(low=np.array([-10]), high=np.array([10]), dtype=np.float32)

        # Initialize state
        self.state = 0

        # Matplotlib setup
        self.fig, self.ax = plt.subplots()
        self.agent, = self.ax.plot([], [], 'bo', markersize=10)  # Agent position
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-1, 1)
        self.ax.axhline(0, color='gray', linewidth=0.5)

    def reset(self):
        self.state = 0  # Reset state to 0
        return np.array([self.state], dtype=np.float32), {}  # Observation and info

    def step(self, action):
        if action == 0:  # Move left
            self.state -= 1
        elif action == 1:  # Move right
            self.state += 1

        # Clip the state to be within [-10, 10]
        self.state = np.clip(self.state, -10, 10)

        # Reward logic
        reward = 1 if self.state == 0 else -0.1
        done = self.state == -10 or self.state == 10  # Episode ends if at boundaries

        return np.array([self.state], dtype=np.float32), reward, done, False, {}

    def render(self, mode='human'):
        if mode == 'human':
            # Update Matplotlib plot
            self.agent.set_data([self.state], [0])  # Wrap self.state in a list
            self.fig.canvas.draw()
            plt.pause(0.01)  # Pause to create animation effect
        else:
            raise NotImplementedError("Render mode not supported")

    def close(self):
        plt.close(self.fig)


if __name__ == "__main__":
    env = CustomEnv()
    obs, _ = env.reset()

    for _ in range(50):  # Run for 50 steps
        action = env.action_space.sample()  # Random action
        obs, reward, done, _, _ = env.step(action)
        print(f"Obs: {obs}, Reward: {reward}, Done: {done}")
        env.render()

        if done:
            break

    env.close()