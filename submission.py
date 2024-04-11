import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import itertools

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register
import frozen_lake
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.utils.plots import Plots
from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper

import frozen_lake_functions
import blackjack_functions
from planner import Planner
from rl import RL

np.random.seed(100)

##################################################################################################################
# Small Frozen Lake (8x8)                                                                                        #
##################################################################################################################
# register(id='CustomFrozenLakeEnv',
#          entry_point='frozen_lake:CustomFrozenLakeEnv',
#          max_episode_steps=20000,
#          kwargs={'map_name': '8x8','is_slippery': True, 'render_mode': None})
# env = gym.make('CustomFrozenLakeEnv')
# map_size = (8, 8)
# actions = {0: '←', 1: '↓', 2: '→', 3: '↑'}
#  
# # Perform Hyperparameter Tuning For VI
# gamma_values = [0.8, 0.85, 0.9, 0.95, 0.99]
# n_iters_values = np.arange(100, 1100, 100)
# theta_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
# params = {'Gamma': gamma_values, 'N-Iters': n_iters_values, 'Theta': theta_values}
# best_params_vi = frozen_lake_functions.perform_hyperparameter_tuning(env, params, 'Small/VI')
# 
# # Perform Value Iteration
# V, V_track, pi, pi_track = Planner(env.P).value_iteration(gamma=0.99, n_iters=600, theta=1e-6)
# 
# frozen_lake_functions.values_heat_map(V, 'Frozen Lake\nValue Iteration State Values', map_size, 'Small/VI')
# frozen_lake_functions.values_per_iteration(V_track, 'Frozen Lake\nValue vs Iteration', 'Small/VI')
# frozen_lake_functions.reward_per_iteration(env, pi_track, 'Small/VI')
# val_max, policy_map = Plots.get_policy_map(pi, V, actions, map_size)
# frozen_lake_functions.plot_policy(val_max, policy_map, 'Frozen Lake\nMapped Policy', map_size, 'Small/VI')
# 
# # Perform Hyperparameter Tuning For PI
# gamma_values = [0.8, 0.85, 0.9, 0.95, 0.99]
# n_iters_values = np.arange(10, 110, 10)
# theta_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
# params = {'Gamma': gamma_values, 'N-Iters': n_iters_values, 'Theta': theta_values}
# best_params_pi = frozen_lake_functions.perform_hyperparameter_tuning(env, params, 'Small/PI')
# 
# # Perform Policy Iteration
# V, V_track, pi, pi_track = Planner(env.P).policy_iteration(gamma=0.99, n_iters=50, theta=1e-8)
# 
# frozen_lake_functions.values_heat_map(V, 'Frozen Lake\Policy Iteration State Values', map_size, 'Small/PI')
# frozen_lake_functions.values_per_iteration(V_track, 'Frozen Lake\nValue vs Iteration', 'Small/PI')
# frozen_lake_functions.reward_per_iteration(env, pi_track, 'Small/PI')
# val_max, policy_map = Plots.get_policy_map(pi, V, actions, map_size)
# frozen_lake_functions.plot_policy(val_max, policy_map, 'Frozen Lake\nMapped Policy', map_size, 'Small/PI')
# 
# # Perform Hyperparameter Tuning for Q-Learning
# gamma_values = [0.9, 0.95, 0.99]
# alpha_decay_values = [0.3, 0.4, 0.5]
# epsilon_decay_values = [0.9, 0.95, 0.99]
# n_episodes_values = [10000, 15000, 200000]
# params = {'Gamma': gamma_values, 'Epsilon Decay': epsilon_decay_values, 'Alpha Decay': alpha_decay_values, 'N-Episodes': n_episodes_values}
# best_params_q = frozen_lake_functions.perform_hyperparameter_tuning(env, params, 'Small/Q-Learning')
# 
# # Perform Q-Learning
# Q, V, pi, Q_track, pi_track, min_track = RL(env).q_learning(gamma=0.99, alpha_decay_ratio=0.5, epsilon_decay_ratio=0.9, n_episodes=50000)
# frozen_lake_functions.values_heat_map(V, 'Frozen Lake\nValue Iteration State Values', map_size, 'Small/Q-Learning')
# val_max, policy_map = Plots.get_policy_map(pi, V, actions, map_size)
# frozen_lake_functions.plot_policy(val_max, policy_map, 'Frozen Lake\nMapped Policy', map_size, 'Small/Q-Learning')

##################################################################################################################
# Large Frozen Lake (20x20)                                                                                      #
##################################################################################################################
# env = gym.make('FrozenLake-v1', desc=generate_random_map(size=20, p=0.8), is_slippery=False, render_mode='human')
# state, info = env.reset()
# state = env.render()
# next_state, reward, terminated, truncated, info = env.step(1)

size = 20
register(id='CustomFrozenLakeEnv',
         entry_point='frozen_lake:CustomFrozenLakeEnv',
         max_episode_steps=1000,
         kwargs={'desc': frozen_lake.generate_random_map(size=size, p=0.8),'is_slippery': True, 'render_mode': None})
env = gym.make('CustomFrozenLakeEnv')
map_size = (size, size)
actions = {0: '←', 1: '↓', 2: '→', 3: '↑'}

# Perform Hyperparameter Tuning For VI
gamma_values = [0.8, 0.85, 0.9, 0.95, 0.99]
n_iters_values = np.arange(100, 1100, 100)
theta_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
params = {'Gamma': gamma_values, 'N-Iters': n_iters_values, 'Theta': theta_values}
best_params_vi = frozen_lake_functions.perform_hyperparameter_tuning(env, params, 'Large/VI')

# Perform Value Iteration
V, V_track, pi, pi_track = Planner(env.P).value_iteration(gamma=0.95, n_iters=400, theta=1e-8)

# Plot Results
frozen_lake_functions.values_heat_map(V, 'Frozen Lake\nValue Iteration State Values', map_size, 'Large/VI')
frozen_lake_functions.values_per_iteration(V_track, 'Frozen Lake\nValue vs Iteration', 'Large/VI')
frozen_lake_functions.reward_per_iteration(env, pi_track, 'Large/VI')
val_max, policy_map = Plots.get_policy_map(pi, V, actions, map_size)
frozen_lake_functions.plot_policy(val_max, policy_map, 'Frozen Lake\nValue Iteration Mapped Policy', map_size, 'Large/VI')

# Perform Hyperparameter Tuning For PI
gamma_values = [0.8, 0.85, 0.9, 0.95, 0.99]
n_iters_values = np.arange(10, 110, 10)
theta_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
params = {'Gamma': gamma_values, 'N-Iters': n_iters_values, 'Theta': theta_values}
best_params_pi = frozen_lake_functions.perform_hyperparameter_tuning(env, params, 'Large/PI')

# Perform Policy Iteration
V, V_track, pi, pi_track = Planner(env.P).policy_iteration(gamma=0.95, n_iters=50, theta=1e-10)

# Plot Results
frozen_lake_functions.values_heat_map(V, 'Frozen Lake\nPolicy Iteration State Values', map_size, 'Large/PI')
frozen_lake_functions.values_per_iteration(V_track, 'Frozen Lake\nValue vs Iteration', 'Large/PI')
frozen_lake_functions.reward_per_iteration(env, pi_track, 'Large/PI')
val_max, policy_map = Plots.get_policy_map(pi, V, actions, map_size)
frozen_lake_functions.plot_policy(val_max, policy_map, 'Frozen Lake\nPolicy Iteration Mapped Policy', map_size, 'Large/PI')

# Perform Hyperparameter Tuning for Q-Learning
gamma_values = [0.2, 0.4, 0.6, 0.8, 0.99]
epsilon_decay_values = [0.2, 0.4, 0.6, 0.8, 0.99]
alpha_decay_values = [0.2, 0.4, 0.6, 0.8, 0.99]
n_episodes_values = [20000, 40000, 60000, 80000, 100000]
params = {'Gamma': gamma_values, 'Epsilon Decay': epsilon_decay_values, 'Alpha Decay': alpha_decay_values, 'N-Episodes': n_episodes_values}
best_params_q = frozen_lake_functions.perform_hyperparameter_tuning(env, params, 'Large/Q-Learning')

# Perform Q-Learning
Q, V, pi, Q_track, pi_track = RL(env).q_learning(gamma=0.95, alpha_decay_ratio=0.4, epsilon_decay_ratio=0.2, n_episodes=100000)
frozen_lake_functions.values_heat_map(V, 'Frozen Lake\nQ-Learning State Values', map_size, 'Large/Q-Learning')
val_max, policy_map = Plots.get_policy_map(pi, V, actions, map_size)
frozen_lake_functions.plot_policy(val_max, policy_map, 'Frozen Lake\nQ-Learning Mapped Policy', map_size, 'Large/Q-Learning')

##################################################################################################################
# Blackjack                                                                                                      #
##################################################################################################################
# env = gym.make('Blackjack-v1', render_mode='human')
# state, info = env.reset()
# state = env.render()

env = gym.make('Blackjack-v1', render_mode=None)
blackjack = BlackjackWrapper(env)
map_size = (29, 10)
actions = {0: 'S', 1: 'H'}

# Perform Hyperparameter Tuning for VI
gamma_values = [0.2, 0.4, 0.6, 0.8, 0.99]
n_iters_values = np.arange(10, 110, 10)
theta_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
params = {'Gamma': gamma_values, 'N-Iters': n_iters_values, 'Theta': theta_values}
best_params_vi = blackjack_functions.perform_hyperparameter_tuning(blackjack, params, 'VI')

# Perform Value Iteration
V, V_track, pi, pi_track = Planner(blackjack.P).value_iteration(gamma=0.6, n_iters=40, theta=1e-6)
blackjack_functions.values_heat_map(V, 'Blackjack\nValue Iteration State Values', map_size, 'VI')
blackjack_functions.values_per_iteration(V_track, 'Blackjack\Value vs Iteration', 'VI')
blackjack_functions.reward_per_iteration(blackjack, pi_track, 'VI')
val_max, policy_map = Plots.get_policy_map(pi, V, actions, map_size)
blackjack_functions.plot_policy(val_max, policy_map, 'Blackjack\nValue Iteration Mapped Policy', 'VI')

# Perform Hyperparameter Tuning for PI
gamma_values = [0.2, 0.4, 0.6, 0.8, 0.99]
n_iters_values = np.arange(5, 55, 5)
theta_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
params = {'Gamma': gamma_values, 'N-Iters': n_iters_values, 'Theta': theta_values}
best_params_pi = blackjack_functions.perform_hyperparameter_tuning(blackjack, params, 'PI')

# Perform Policy Iteration
V, V_track, pi, pi_track = Planner(blackjack.P).policy_iteration(gamma=0.8, n_iters=25, theta=1e-7)
blackjack_functions.values_heat_map(V, 'Blackjack\nPolicy Iteration State Values', map_size, 'PI')
blackjack_functions.values_per_iteration(V_track, 'Blackjack\nValue vs Iteration', 'PI')
blackjack_functions.reward_per_iteration(blackjack, pi_track, 'PI')
val_max, policy_map = Plots.get_policy_map(pi, V, actions, map_size)
blackjack_functions.plot_policy(val_max, policy_map, 'Blackjack\nPolicy Iteration Mapped Policy', 'PI')

# Perform Hyperparameter Tuning for Q-Learning
gamma_values = [0.2, 0.4, 0.6, 0.8, 0.99]
epsilon_decay_values = [0.2, 0.4, 0.6, 0.8, 0.99]
alpha_decay_values = [0.2, 0.4, 0.6, 0.8, 0.99]
n_episodes_values = [20000, 40000, 60000, 80000, 100000]
params = {'Gamma': gamma_values, 'Epsilon Decay': epsilon_decay_values, 'Alpha Decay': alpha_decay_values, 'N-Episodes': n_episodes_values}
best_params_q = blackjack_functions.perform_hyperparameter_tuning(blackjack, params, 'Q-Learning')

# Perform Q-Learning
Q, V, pi, Q_track, pi_track = RL(blackjack).q_learning(gamma=0.4, alpha_decay_ratio=0.99, epsilon_decay_ratio=0.99, n_episodes=100000)
blackjack_functions.values_heat_map(V, 'Blackjack\nQ-Learning State Values', map_size, 'Q-Learning')
val_max, policy_map = Plots.get_policy_map(pi, V, actions, map_size)
blackjack_functions.plot_policy(val_max, policy_map, 'Blackjack\nQ-Learning Mapped Policy', 'Q-Learning')