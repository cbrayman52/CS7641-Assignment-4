import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import itertools

import gymnasium as gym
from bettermdptools.algorithms.planner import Planner
from bettermdptools.algorithms.rl import RL
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.utils.plots import Plots
from bettermdptools.utils.grid_search import GridSearch

def perform_hyperparameter_tuning(env, params, alg):

    # List to store hyperparameters
    hyperparameters = []

    param_keys = list(params.keys())

    for i in itertools.product(*params.values()):
        param_dict = dict(zip(param_keys, i))

        if 'VI' in alg:
            _, _, pi = Planner(env.P).value_iteration(gamma=param_dict['Gamma'], n_iters=param_dict['N-Iters'], theta=param_dict['Theta'])
            title = 'Value Iteration Score '
        elif 'PI' in alg:
            _, _, pi = Planner(env.P).policy_iteration(gamma=param_dict['Gamma'], n_iters=param_dict['N-Iters'], theta=param_dict['Theta'])
            title = 'Policy Iteration Score '
        else:
            _, _, pi, _, _ = RL(env).q_learning(gamma=param_dict['Gamma'], epsilon_decay_ratio=param_dict['Epsilon Decay'], n_episodes=param_dict['N-Episodes'])
            title = 'Q-Learning Score '

        test_scores = np.mean(TestEnv.test_env(env=env, n_iters=1000, pi=pi))
        hyperparameters.append({**param_dict, 'Score':test_scores})

    # Create DataFrame
    grid_search_results = pd.DataFrame(hyperparameters)

    # Create subplots
    fig, axs = plt.subplots(1, len(param_keys), figsize=(5*len(param_keys), 5), sharey=True)

    # Iterate through parameter keys and plot corresponding data
    for i, param_key in enumerate(param_keys):
        sns.barplot(data=grid_search_results[[param_key, 'Score']], x=param_key, y='Score', ax=axs[i], palette=sns.color_palette("tab10"))
        axs[i].set_title(title + param_key)
        axs[i].set_xlabel(param_key)
        axs[i].set_ylabel('')
        axs[i].grid(False)

        # Set x-tick labels for 'Theta' parameter
        if param_key == 'Theta':
            theta_values = ['{:.0e}'.format(theta) for theta in params['Theta']]
            theta_labels = [str(theta) for theta in theta_values]
            axs[i].set_xticklabels(theta_labels)

    axs[0].set_ylabel(title)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f'Images/Blackjack/{alg}/Hyperparameter Tuning.png')
    plt.close()

    # Return the best parameters
    max_value_index = grid_search_results['Score'].idxmax()
    highest_value = grid_search_results.loc[max_value_index, 'Score']
    rows_with_highest_value = grid_search_results[grid_search_results['Score'] == highest_value]

    return rows_with_highest_value

def values_heat_map(data, title, size, alg):
    data = np.around(np.array(data).reshape(size), 2)
    df = pd.DataFrame(data=data)

    plt.figure(figsize=(10,10))
    sns.heatmap(df, 
                annot=True,
                fmt='',
                cmap=sns.color_palette('magma_r', as_cmap=True),
                linewidths=0.7,
                linecolor='black',
                yticklabels=['H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21',
                             'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'BJ'],
                xticklabels=['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A'],)
    plt.title(title, fontsize=20)
    plt.savefig(f'Images/Blackjack/{alg}/Value Heat Map.png')
    plt.close()
    return

def values_per_iteration(data, title, alg):
    mean_value_per_iter = np.trim_zeros(np.mean(data, axis=1), 'b')
    max_value_per_iter = np.trim_zeros(np.max(data, axis=1), 'b')
    mean_df = pd.DataFrame(data=mean_value_per_iter)
    max_df = pd.DataFrame(data=max_value_per_iter)

    sns.set_theme(style='whitegrid')
    ax = sns.lineplot(data=mean_df, legend=False, palette=['blue'], label='Mean Value')
    sns.lineplot(data=max_df, legend=False, ax=ax, palette=['orange'], label='Max Value')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'Images/Blackjack/{alg}/Value vs Iteration.png')
    plt.close()
    return

def reward_per_iteration(env, pi_track, alg):
    mean_test_scores_list = []
    episodes_list = []

    for episode, pi in enumerate(pi_track):
        test_scores = TestEnv.test_env(env=env, n_iters=1000, pi=pi)
        mean_test_scores = np.mean(test_scores)
        mean_test_scores_list.append(mean_test_scores)
        episodes_list.append(episode)
        
    plt.plot(episodes_list, mean_test_scores_list)
    plt.title('Reward vs Iteration')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Test Scores')
    plt.grid(True)
    plt.savefig(f'Images/Blackjack/{alg}/Reward vs Iteration.png')
    plt.close()
    return

def plot_policy(val_max, actions, title, alg):
    plt.figure(figsize=(10,10))
    sns.heatmap(
        val_max,
        annot=actions,
        fmt='',
        cmap=sns.color_palette('magma_r', as_cmap=True),
        linewidths=0.7,
        linecolor='black',
        yticklabels=['H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21',
                     'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'BJ'],
        xticklabels=['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A'],
    )
    plt.title(title, fontsize=20)
    plt.savefig(f'Images/Blackjack/{alg}/Policy Map.png')
    plt.close()
    return