import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

import gymnasium as gym
from bettermdptools.algorithms.rl import RL
from bettermdptools.algorithms.planner import Planner
# from bettermdptools.utils.test_env import TestEnv
from bettermdptools.utils.plots import Plots
from test_env import TestEnv

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
            _, _, pi, _, _= RL(env).q_learning(gamma=param_dict['Gamma'], epsilon_decay_ratio=param_dict['Epsilon Decay'], n_episodes=param_dict['N-Episodes'])
            title = 'Q-Learning Score '

        test_scores = np.mean(TestEnv.test_env(env=env, n_iters=10, pi=pi))
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
    plt.savefig(f'Images/Frozen Lake/{alg}/Hyperparameter Tuning.png')
    plt.close()

    # Return the best parameters
    max_value_index = grid_search_results['Score'].idxmax()
    highest_value = grid_search_results.loc[max_value_index, 'Score']
    rows_with_highest_value = grid_search_results[grid_search_results['Score'] == highest_value]

    return rows_with_highest_value

def values_heat_map(data, title, size, alg):

    data = np.around(np.array(data).reshape(size), 2)
    df = pd.DataFrame(data=data)

    # Create a mask where cells with value 0 are marked as True
    mask = data == 0.
    mask[-1][-1] = False
    mask[19][9] = False
    cmap = sns.color_palette('Blues', as_cmap=True)
    cmap.set_bad("grey") 
    
    plt.figure(figsize=size)
    sns.heatmap(
        df,
        annot=True,
        fmt='',
        cmap=cmap,
        center=0,
        linewidths=0.7,
        linecolor='black',
        mask=mask,
        annot_kws={'fontsize': 'large'},
        )
    plt.title(title, fontsize=40)
    plt.savefig(f'Images/Frozen Lake/{alg}/Value Heat Map.png')
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
    plt.savefig(f'Images/Frozen Lake/{alg}/Value vs Iteration.png')
    plt.close()
    return

def reward_per_iteration(env, pi_track, alg):
    mean_test_scores_list = []
    episodes_list = []
    mean_test_scores_sum = 0

    if 'PI' in alg:
        for episode, pi in enumerate(pi_track):
            test_scores = TestEnv.test_env(env=env, n_iters=10, pi=pi)
            mean_test_scores = np.mean(test_scores)
            mean_test_scores_list.append(mean_test_scores)
            episodes_list.append(episode)
    else:
        for episode, pi in enumerate(pi_track):
            test_scores = TestEnv.test_env(env=env, n_iters=10, pi=pi)
            mean_test_scores = np.mean(test_scores)
            mean_test_scores_sum += mean_test_scores
            if episode % 10 == 9:
                mean_test_scores_list.append(mean_test_scores_sum)
                mean_test_scores_sum = 0
                episodes_list.append(episode)
        
    plt.plot(episodes_list, mean_test_scores_list)
    plt.title('Reward vs Iteration')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Test Scores')
    plt.grid(True)
    plt.savefig(f'Images/Frozen Lake/{alg}/Reward vs Iteration.png')
    plt.close()
    return

def plot_policy(val_max, directions, title, size, alg):
    if size == (20,20):
        plt.figure(figsize=(10,10))
    else:
        plt.figure(figsize=size)

    # Create a mask where cells with value 0 are marked as True
    mask = val_max == 0.
    mask[-1][-1] = False
    cmap = sns.color_palette('Blues', as_cmap=True)
    cmap.set_bad("grey") 
   
    sns.heatmap(
        val_max,
        annot=directions,
        fmt='',
        cmap=cmap,
        linewidths=0.7,
        linecolor='black',
        xticklabels=[],
        yticklabels=[],
        mask=mask,
        annot_kws={'fontsize': 'large'},
        )
    plt.title(title, fontsize=20)
    plt.savefig(f'Images/Frozen Lake/{alg}/Policy Map.png')
    plt.close()
    return

def reward_per_iteration_q(env, pi_track, vertical_lines, alg):
    mean_test_scores_list = []
    episodes_list = []
    cum_sum_rewards = []
    
    for episode, pi in enumerate(pi_track):
        test_scores = TestEnv.test_env(env=env, n_iters=10, pi=pi)
        mean_test_scores = np.mean(test_scores)
        mean_test_scores_list.append(mean_test_scores)
        episodes_list.append(episode)
        
        cum_sum_reward += mean_test_scores
        
        if (episode + 1) % 100 == 0:
            cum_sum_rewards.append(cum_sum_reward)
            cum_sum_reward = 0
            
            plt.plot(range(0, episode + 1, 100), cum_sum_rewards)
    
    plt.title('Cumulative Reward vs Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)

    ymin, ymax = plt.ylim()  # Get the y-axis limits
    
    # Adding vertical dashed lines at specific x-labels
    for label, x_label in vertical_lines.items():
        plt.axvline(x=x_label, color='r', linestyle='--', label='_nolegend_')
        plt.text(x_label, ymin - 0.01 * (ymax - ymin), label, ha='center', va='top')

    plt.savefig(f'Images/Frozen Lake/{alg}/Cumulative Reward vs Episode.png')
    plt.close()
    return