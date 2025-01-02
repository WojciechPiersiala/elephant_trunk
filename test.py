#!/home/wp/Studia/soft_robotics/elephant_trunk/trunk/bin/python
# Enable Interactive Plots
#%matplotlib inline
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt
import os
import shutil
from manipulator.trunk_environment import TrunkEnv  # Import TrunkEnv


class TrunkAgent:
    def __init__(self, env: gym.Env, learning_rate: float, epsilon: float, epsilon_decay: float, final_epsilon: float, discount_factor: float = 0.95):
        """Initialize a reinforcement learning agent."""
        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

        # Discretize the observation space
        N_BINS = 20
        self.obs_bins = [
            np.linspace(-20, 20, N_BINS),  # x-effector
            np.linspace(-30, 0, N_BINS),   # y-effector
            np.linspace(-20, 20, N_BINS),  # x-target
            np.linspace(-30, 0, N_BINS)    # y-target
        ]

        # Discretize the action space
        ACTION_BINS = 20
        self.action_bins = [
            np.linspace(self.env.action_space.low[i], self.env.action_space.high[i], ACTION_BINS)
            for i in range(self.env.action_space.shape[0])
        ]

        # Initialize Q-table
        self.q_values = defaultdict(lambda: np.zeros(ACTION_BINS ** self.env.action_space.shape[0]))

    def discretize_observation(self, obs):
        """Discretizes the continuous observation into bins."""
        discrete_obs = tuple(np.digitize(obs[i], self.obs_bins[i]) - 1 for i in range(len(obs)))
        return discrete_obs

    def discretize_action(self, action):
        """Discretizes a continuous action into bins."""
        discrete_action = tuple(np.digitize(action[i], self.action_bins[i]) - 1 for i in range(len(action)))
        return discrete_action

    def get_action(self, obs):
        """Returns the best action or a random one based on epsilon."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            obs_tuple = self.discretize_observation(obs)
            best_action_idx = np.argmax(self.q_values[obs_tuple])

            # Map back to continuous action space
            best_action = np.array([
                self.action_bins[i][best_action_idx % len(self.action_bins[i])]
                for i in range(len(self.action_bins))
            ])
            return best_action

    def update(self, obs, action, reward, terminated, next_obs):
        """Updates a Q-value of an action."""
        tp_obs = self.discretize_observation(obs)
        tp_action = self.discretize_action(action)
        action_idx = np.ravel_multi_index(tp_action, [len(b) for b in self.action_bins])

        tp_next_obs = self.discretize_observation(next_obs)
        future_q_value = (not terminated) * np.max(self.q_values[tp_next_obs])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[tp_obs][action_idx]
        self.q_values[tp_obs][action_idx] += self.lr * temporal_difference
        self.training_error.append(float(temporal_difference))


    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.final_epsilon)

# Hyperparameters
learning_rate = 0.01
n_episodes = 5000
start_epsilon = 1.0
final_epsilon = 0.05
epsilon_decay = start_epsilon / (n_episodes)

# Path to the folder
folder_path = "./trunk-agent"

# Clear the folder if it exists
if os.path.exists(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove files
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directories
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
else:
    os.makedirs(folder_path)  # Create folder if it doesn't exist

# Environment setup
env = gym.make("TrunkManipulator-v0", render_mode="rgb_array", max_steps=100)
env = gym.wrappers.RecordVideo(env, video_folder="trunk-agent", name_prefix="eval", episode_trigger=lambda x: x == n_episodes or x == 0)
env = gym.wrappers.RecordEpisodeStatistics(env=env)
agent = TrunkAgent(env=env, learning_rate=learning_rate, epsilon=start_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)

# Training loop
episode_td_errors = []
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    episode_td_error = []
    while not done:
        action = agent.get_action(obs)
        # print(f"action : {action}")
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(action=action, obs=obs, next_obs=next_obs, reward=reward, terminated=terminated)
        obs = next_obs
        done = terminated or truncated
        episode_td_error.append(agent.training_error[-1])

        # if episode % 100 == 0:
        #     print(f"Episode {episode}, Reward: {sum(env.return_queue)}, Epsilon: {agent.epsilon}")
    episode_td_errors.append(np.mean(episode_td_error))
    agent.decay_epsilon()

# Plotting the training error
plt.close('all')
rolling_mean = np.convolve(episode_td_errors, np.ones(1) / 1, mode='valid')
fig1, ax = plt.subplots(3,1,figsize=(10, 12))
ax[0].plot(rolling_mean)
ax[0].set_title("Training Error")
ax[0].set_xlabel("Episode")
ax[0].set_ylabel("Mean Temporal Difference")

ax[1].plot(env.return_queue)
ax[1].set_title("Episode Rewards")
ax[1].set_xlabel("Episode")
ax[1].set_ylabel("Reward")

ax[2].plot(env.length_queue)
ax[2].set_title("Episode Lengths")
ax[2].set_xlabel("Episode")
ax[2].set_ylabel("Length")

plt.tight_layout()
plt.show()
env.close()