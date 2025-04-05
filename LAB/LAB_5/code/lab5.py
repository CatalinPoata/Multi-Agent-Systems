import ast
import os

import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import json


EPSILONS = [0.1, 0.5, 0.9]
GAMMAS = [0.5, 0.9]
LEARNING_RATES = [0.1, 0.5, 0.9]
EPISODES = 50000
CHECKPOINT_EVERY = 50
EXPERIMENT_TRIES = 50
MAX_STEPS = 1000
FIG_SIZE = (10, 6)

def epsilon_greedy(env, s, q, epsilon):
    prob = np.random.rand()
    if prob < epsilon:
        # print('Exploration')
        return env.action_space.sample()
    else:
        # print("Exploitation")
        return np.argmax([q[s, a] for a in range(env.action_space.n)])


def cosine_annealing(episode, total_episodes, epsilon_max=1.0, epsilon_min=0.01):
    return epsilon_min + 0.5 * (epsilon_max - epsilon_min) * (1 + np.cos(np.pi * episode / total_episodes))



def q_learning(env, gamma, epsilon, learning_rate):
    states = env.observation_space.n
    actions = env.action_space.n

    q = np.zeros((states, actions))

    # print(q)
    # print(q.shape)
    average_rewards = []

    for episode in range(EPISODES):
        # print('Episode:', episode)
        curr_epsilon = cosine_annealing(episode, EPISODES, epsilon)
        state, _ = env.reset()
        terminated = False
        cur_steps = 0
        while not terminated and cur_steps < MAX_STEPS:
            action = epsilon_greedy(env, state, q, curr_epsilon)
            next_state, reward, terminated, _, _= env.step(action)
            q[state, action] = q[state, action] + learning_rate * (reward + gamma * max(q[next_state, :]) - q[state, action])
            state = next_state
            cur_steps += 1

        if episode % CHECKPOINT_EVERY == 0:
            # print(f"Current epsilon: {curr_epsilon}")
            sum_rewards = 0
            policy = np.zeros(states)
            for s in range(states):
                policy[s] = np.argmax(q[s, :])

            for run in range(EXPERIMENT_TRIES):
                # print(f"Start run: {run}")
                run_state, _ = env.reset()
                terminated_run = False
                run_steps = 0
                while not terminated_run and run_steps < MAX_STEPS:
                    run_action = policy[run_state]
                    next_run_state, run_reward, terminated_run, _, _ = env.step(run_action)
                    sum_rewards += run_reward
                    run_state = next_run_state
                    run_steps += 1
                # print(f"Finish run: {run}")

            average_reward = sum_rewards / EXPERIMENT_TRIES
            average_rewards.append(average_reward)

        # print(f"Done episode {episode}")
    pi = np.zeros(states)
    for s in range(states):
        pi[s] = np.argmax(q[s, :])
    return pi, average_rewards

def sarsa(env, gamma, epsilon, learning_rate):
    states = env.observation_space.n
    actions = env.action_space.n

    q = np.zeros((states, actions))
    # print(q)
    # print(q.shape)
    average_rewards = []
    for episode in range(EPISODES):
        # print('Episode:', episode)
        curr_epsilon = cosine_annealing(episode, EPISODES, epsilon)
        state, _ = env.reset()
        terminated = False
        curr_steps = 0
        action = epsilon_greedy(env, state, q, curr_epsilon)
        while not terminated and curr_steps < MAX_STEPS:
            next_state, reward, terminated, _, _ = env.step(action)
            action_prime = epsilon_greedy(env, next_state, q, curr_epsilon)
            q[state, action] = q[state, action] + learning_rate * (reward + gamma * q[next_state, action_prime] - q[state, action])
            state = next_state
            action = action_prime
            curr_steps += 1

        if episode % CHECKPOINT_EVERY == 0:
            # print(f"Current epsilon: {curr_epsilon}")
            sum_rewards = 0
            policy = np.zeros(states)
            for s in range(states):
                policy[s] = np.argmax(q[s, :])

            for run in range(EXPERIMENT_TRIES):
                run_state, _ = env.reset()
                terminated_run = False
                run_steps = 0
                while not terminated_run and run_steps < MAX_STEPS:
                    run_action = policy[run_state]
                    next_run_state, run_reward, terminated_run, _, _ = env.step(run_action)
                    sum_rewards += run_reward
                    run_state = next_run_state
                    run_steps += 1

            average_reward = sum_rewards / EXPERIMENT_TRIES
            average_rewards.append(average_reward)
        # print(f"Done episode {episode}")

    pi = np.zeros(states)
    for s in range(states):
        pi[s] = np.argmax(q[s, :])

    return pi, average_rewards

def compute_data():
    envs = [gymnasium.make('Taxi-v3'), gymnasium.make('FrozenLake-v1')]
    results = {}
    for env in envs:
        for gamma in GAMMAS:
            for learning_rate in LEARNING_RATES:
                for epsilon in EPSILONS:
                    print(f"Running experiment for env={env.spec.id}, gamma={gamma}, learning_rate={learning_rate}, epsilon={epsilon}")
                    q_policy, q_rewards = q_learning(env, gamma, epsilon, learning_rate)
                    s_policy, s_rewards = sarsa(env, gamma, epsilon, learning_rate)
                    key = (env.spec.id, gamma, learning_rate, epsilon).__str__()

                    results[key] = {
                        'q_rewards': q_rewards,
                        's_rewards': s_rewards,
                        'q_policy': q_policy.tolist(),
                        's_policy': s_policy.tolist(),
                        'q_max_reward': np.max(q_rewards).__str__(),
                        's_max_reward': np.max(s_rewards).__str__()
                    }
    return results


def run_experiments():
    # data = compute_data()
    # with open('../results/results.txt', 'w') as f:
    #     json.dump(data, f, indent=4)

    with open('../results/results.txt', 'r+') as f:
        data = json.load(f)

    envs = [gymnasium.make('Taxi-v3'), gymnasium.make('FrozenLake-v1')]
    for env in envs:
        for gamma in GAMMAS:
            for learning_rate in LEARNING_RATES:
                plt.figure(figsize=FIG_SIZE)
                for epsilon in EPSILONS:
                    key = (env.spec.id, gamma, learning_rate, epsilon).__str__()
                    q_rewards = data[key]['q_rewards']
                    s_rewards = data[key]['s_rewards']
                    max_q = data[key]['q_max_reward']
                    max_s = data[key]['s_max_reward']
                    index_max_q = q_rewards.index(float(max_q))
                    index_max_s = s_rewards.index(float(max_s))
                    plt.plot(q_rewards, label=f'Q-Learning Rewards for Epsilon = {epsilon}', alpha=0.6)
                    plt.plot(s_rewards, label=f'SARSA Rewards for Epsilon = {epsilon}', alpha=0.6)
                    plt.scatter(index_max_q, float(max_q), marker='o',
                                label=f'Max Q-Learning Reward for Epsilon = {epsilon}')
                    plt.scatter(index_max_s, float(max_s), marker='o',
                                label=f'Max SARSA Reward for Epsilon = {epsilon}')
                plt.xlabel('Checkpoint')
                plt.ylabel('Reward')
                plt.title(f"γ={gamma}, α={learning_rate}")
                plt.legend()
                os.makedirs(f"../results/{ast.literal_eval(key)[0]}", exist_ok=True)
                plt.savefig(f"../results/{ast.literal_eval(key)[0]}/G{GAMMAS.index(gamma)}_LR{LEARNING_RATES.index(learning_rate)}.png")
                plt.close()

    for env in envs:
        for epsilon in EPSILONS:
            for learning_rate in LEARNING_RATES:
                plt.figure(figsize=FIG_SIZE)
                for gamma in GAMMAS:
                    key = (env.spec.id, gamma, learning_rate, epsilon).__str__()
                    q_rewards = data[key]['q_rewards']
                    s_rewards = data[key]['s_rewards']
                    max_q = data[key]['q_max_reward']
                    max_s = data[key]['s_max_reward']
                    index_max_q = q_rewards.index(float(max_q))
                    index_max_s = s_rewards.index(float(max_s))
                    plt.plot(q_rewards, label=f'Q-Learning Rewards for Gamma = {gamma}', alpha=0.6)
                    plt.plot(s_rewards, label=f'SARSA Rewards for Gamma = {gamma}', alpha=0.6)
                    plt.scatter(index_max_q, float(max_q), marker='o',
                                label=f'Max Q-Learning Reward for Gamma = {gamma}')
                    plt.scatter(index_max_s, float(max_s), marker='o',
                                label=f'Max SARSA Reward for Gamma = {gamma}')
                plt.xlabel('Checkpoint')
                plt.ylabel('Reward')
                plt.title(f"ε={epsilon}, α={learning_rate}")
                plt.legend()
                os.makedirs(f"../results/{ast.literal_eval(key)[0]}", exist_ok=True)
                plt.savefig(
                    f"../results/{ast.literal_eval(key)[0]}/E{EPSILONS.index(epsilon)}_LR{LEARNING_RATES.index(learning_rate)}.png")
                plt.close()

    for env in envs:
        for epsilon in EPSILONS:
            for gamma in GAMMAS:
                plt.figure(figsize=FIG_SIZE)
                for learning_rate in LEARNING_RATES:
                    key = (env.spec.id, gamma, learning_rate, epsilon).__str__()
                    q_rewards = data[key]['q_rewards']
                    s_rewards = data[key]['s_rewards']
                    max_q = data[key]['q_max_reward']
                    max_s = data[key]['s_max_reward']
                    index_max_q = q_rewards.index(float(max_q))
                    index_max_s = s_rewards.index(float(max_s))
                    plt.plot(q_rewards, label=f'Q-Learning Rewards for Learning Rate = {learning_rate}', alpha=0.6)
                    plt.plot(s_rewards, label=f'SARSA Rewards for Learning Rate = {learning_rate}', alpha=0.6)
                    plt.scatter(index_max_q, float(max_q), marker='o',
                                label=f'Max Q-Learning Reward for Learning Rate = {learning_rate}')
                    plt.scatter(index_max_s, float(max_s), marker='o',
                                label=f'Max SARSA Reward for Learning Rate = {learning_rate}')
                plt.xlabel('Checkpoint')
                plt.ylabel('Reward')
                plt.title(f"ε={epsilon}, γ={gamma}")
                plt.legend()
                os.makedirs(f"../results/{ast.literal_eval(key)[0]}", exist_ok=True)
                plt.savefig(f"../results/{ast.literal_eval(key)[0]}/E{EPSILONS.index(epsilon)}_G{GAMMAS.index(gamma)}.png")
                plt.close()




if __name__ == '__main__':
    run_experiments()