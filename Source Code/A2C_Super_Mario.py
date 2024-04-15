    # The code is designed for training and testing reinforcement learning agents using the Super Mario Bros 2 environment.
    # It utilizes the Stable Baselines3 library, specifically the Advantage Actor Critic (A2C) algorithm.
    # Command-line arguments are used to specify whether to train or test the agent, the learning algorithm (in this case, A2C), and optionally, a seed number.
    # Hyperparameters such as environment ID, number of training steps, learning rate, and gamma are defined.
    # The code defines a function to create the environment with specific wrappers for preprocessing.
    # A A2C agent is created with custom hyperparameters.
    # Depending on the mode (training or testing), the agent is either trained or a pre-trained policy is loaded.
    # The policy is evaluated using the evaluate_policy function.
    # The learned behavior of the agent is visualized by rendering the environment and printing episode statistics such as steps per episode, reward per episode, and total game score.

#  This code is tested on Ubunutu 22.04 System
#  GPU System configuration = CUDA 11.8 
#  Instructions manual for the code --------
#   Install PyTorch(Here Ubuntu system is using as default) -  pip3 install torch torchvision torchaudio 


#   Install pip install gym==0.21.0
#           pip install swig
#           pip install box2d-py
#           pip install gymnasium
#           pip install pyglet==1.5.11
#           pip install stable-baselines3[extra]==1.8.0
#           pip install gym-super-mario-bros==7.3.0

#  To run the code use this command line - python  A2C_Super_Mario.py (test|train) A2C <seed_number>



import sys  # Importing sys module to access command-line arguments.
import gym  # Importing gym module for creating reinforcement learning environments.
import pickle  # Importing pickle module for serializing and deserializing Python objects.
import random  # Importing random module for generating random numbers.
from typing import Callable  # Importing typing module for type hinting.
from stable_baselines3 import A2C  # Importing A2C algorithm from stable_baselines3 library.
from stable_baselines3.common.evaluation import evaluate_policy  # Importing evaluation functions.

import gym_super_mario_bros  # Importing Super Mario Bros environment from gym_super_mario_bros package.
from nes_py.wrappers import JoypadSpace  # Importing JoypadSpace wrapper from nes_py.wrappers for modifying the action space.
from gym_super_mario_bros.actions import RIGHT_ONLY  # Importing RIGHT_ONLY action set from gym_super_mario_bros.actions for restricting actions.
from stable_baselines3.common import atari_wrappers  # Importing Atari wrappers for pre-processing the environment.



# Check for the correct usage of the script
if len(sys.argv) < 2 or len(sys.argv) > 4:
    print("USAGE: sb-SuperMarioBros2-v1.py (train|test) A2C [seed_number]")
    exit(0)

# Define parameters
environmentID = "SuperMarioBros2-v1"
trainMode = True if sys.argv[1] == 'train' else False
seed = int(sys.argv[3]) if len(sys.argv) == 4 else None
if seed is None:
    seed = random.randint(0, 1000)
policyFileName = "A2C-" + environmentID + "-seed" + str(seed) + ".policy.pkl"
num_training_steps = 2000000
num_test_episodes = 20
learning_rate = 0.00085
gamma = 0.895
policy_rendering = True

# Create the learning environment 
def make_env(gym_id, seed):
    env = gym_super_mario_bros.make(gym_id)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = atari_wrappers.MaxAndSkipEnv(env, 4)
    env = atari_wrappers.NoopResetEnv(env, noop_max=30)
    env = atari_wrappers.ClipRewardEnv(env)
    env.seed(seed)	
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

# Initialize the environment
environment = make_env(environmentID, seed)

# Create the A2C agent's model
model = A2C("CnnPolicy", environment, seed=seed, learning_rate=learning_rate, gamma=gamma, n_steps=10, ent_coef=0.02, verbose=1)

# Train the agent or load a pre-trained one
if trainMode:
    model.learn(total_timesteps=num_training_steps, progress_bar=True)
    print("Saving policy " + str(policyFileName))
    pickle.dump(model.policy, open(policyFileName, 'wb'))
else:
    print("Loading policy...")
    with open(policyFileName, "rb") as f:
        policy = pickle.load(f)
    model.policy = policy
import time
print("Evaluating policy...")
 
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=num_test_episodes)
print("EVALUATION: mean_reward=%s std_reward=%s" % (mean_reward, std_reward))

# Visualize the agent's learned behavior
steps_per_episode = 0
reward_per_episode = 0
total_cumulative_reward = 0
total_game_score = 0  # Initialize total game score
episode = 1
num_episodes = 0  # Initialize number of episodes

env = model.get_env()
obs = env.reset()
start_time = time.time()  # Record start time
while True and policy_rendering:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    steps_per_episode += 1
    reward_per_episode += reward
    
    if any(done):
        score = 0
        score = info[0]['score']
        total_game_score += info[0]['score']
        print("episode=%s, steps_per_episode=%s, reward_per_episode=%s total_game_score=%s" % (episode, steps_per_episode, reward_per_episode,score))
        total_cumulative_reward += reward_per_episode
        steps_per_episode = 0
        reward_per_episode = 0
        episode += 1
        num_episodes += 1  # Increment number of episodes
        obs = env.reset()
    env.render("human")

    if episode > num_test_episodes: 
        end_time = time.time()  # Record end time
        test_time = end_time - start_time  # Calculate total test time
        print("total_cumulative_reward=%s avg_cumulative_reward=%s avg_game_score=%s test_time=%s seconds" % \
              (total_cumulative_reward, total_cumulative_reward/num_test_episodes, total_game_score/num_test_episodes, test_time))
        break

env.close()
