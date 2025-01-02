import gymnasium as gym
from manipulator.manipulator import Manipulator
from gymnasium import spaces
import numpy as np
import random 
import matplotlib.pyplot as plt
import matplotlib


class TrunkEnv(gym.Env):
    metadata = {"render_modes" : ["human", "rgb_array"], 
                "render_fps": 30}
    
    def __init__(self, target :list[float,float],render_mode :str="human", max_steps :int=100):
        super().__init__()

        # matplotlib.use('Agg')  # Use the Agg backend for rendering

        # choose render mode
        assert render_mode in self.metadata["render_modes"], f"Invalid render mode, correct render modes : {self.metadata["render_modes"]}"
        self.render_mode = render_mode

        self.target = target
        self.manipulator = Manipulator(manual_mode=False, target_cor=self.target, render_mode=self.render_mode, max_steps=max_steps)

        # Definition of action space - vector of curvature update values
        SK_STEP = 0.1
        self.action_space = spaces.Box(low = -SK_STEP, high=SK_STEP, shape=(4,), dtype=np.float32)

        # Definition of observation space - coordinates of end the effector and the target
        # [x_effector, y_effector, x_target, y_target] # x ∈ [-20.0, 20], y ∈ [-30, 0] 
        low = np.concatenate((np.array([-20, -30, -20, -30]), np.full(4, -np.inf)))
        high = np.concatenate((np.array([20, 0, 20, 0]), np.full(4, np.inf)))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.max_steps = max_steps

    def reset(self, seed=None, options=None):
        if hasattr(self, 'manipulator') and self.manipulator.fig:
            self.manipulator.close()  # Close the existing manipulator's plot
        # target_x = random.uniform(-20, 20) # target x range
        # target_y = random.uniform(-30, 0)  # target y range
        self.manipulator = Manipulator(manual_mode=False, target_cor=self.target, render_mode=self.render_mode, max_steps=self.max_steps)

        # observation = np.array(self.manipulator.state_space, dtype=np.float32)
        observation = np.concatenate(self.manipulator.state_space, dtype=np.float32)
        return observation, {}
    
    
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action, action space does not contain the action: {action} "
        self.manipulator.run(action_space=action)
        observation = np.concatenate(self.manipulator.state_space, dtype=np.float32)
        
        reward = -self.manipulator.dist_to_target
        # reward = 0
        if self.manipulator.on_target:
            reward += 1000  # Large reward for success
        elif self.manipulator.terminate:
            reward -= 500  # Penalty for truncation

        done = self.manipulator.on_target
        truncated = self.manipulator.terminate

        return observation, reward, done, truncated, {}


    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            plt.pause(0.001)  # Update the matplotlib plot
        elif self.render_mode == "rgb_array":
            # Generate an ARGB array representation
            self.manipulator.fig.canvas.draw()
            argb_image = np.frombuffer(self.manipulator.fig.canvas.tostring_argb(), dtype=np.uint8)
            argb_image = argb_image.reshape((self.manipulator.fig.canvas.get_width_height()[::-1] + (4,)))
            
            # Convert ARGB to RGB by dropping the alpha channel
            rgb_image = argb_image[:, :, 1:]  # Slice to keep only RGB channels
            return rgb_image
        

    def close(self):
        """Clean up resources."""
        plt.close(self.manipulator.fig)

