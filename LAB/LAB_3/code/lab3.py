import gym
import numpy as np
import matplotlib.pyplot as plt
import heapq

GAMMA = 0.9
EPSILON = 1e-3
MAX_ITERATIONS = int(5e5)

def value_iteration(env, gamma=GAMMA, epsilon=EPSILON, max_iterations=MAX_ITERATIONS):
    V = np.random.randn(env.observation_space.n)
    for i in range(max_iterations):
        delta = 0
        V_old = V.copy()
        for s in range(env.observation_space.n):
            v = V[s]
            V[s] = max(sum(p * (r + gamma * V_old[s_]) for p, s_, r, _ in env.P[s][a]) for a in range(env.action_space.n))
            delta = max(delta, abs(v - V[s]))
        if delta < epsilon:
            break
    return V


def gauss_seidel_value_iteration(env, V_star, gamma=GAMMA, epsilon=EPSILON, max_iterations=MAX_ITERATIONS):
    V = np.random.randn(env.observation_space.n)
    errors = []
    iters = []
    for i in range(max_iterations):
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            V[s] = max(sum(p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]) for a in range(env.action_space.n))
            delta = max(delta, abs(v - V[s]))
        errors.append(np.linalg.norm(V - V_star))
        iters.append(i + 1)
        if delta < epsilon:
            break
    return iters, errors


def prioritized_sweeping_vi(env, V_star, gamma=GAMMA, epsilon=EPSILON, max_iterations=MAX_ITERATIONS):
    V = np.random.randn(env.observation_space.n)
    queue = []
    errors = []
    iters = []
    for s in range(env.observation_space.n):
        v = V[s]
        best_v = max(sum(p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]) for a in range(env.action_space.n))
        diff = abs(v - best_v)
        heapq.heappush(queue, (-diff, s))

    for i in range(max_iterations):
        if not queue:
            break
        _, s = heapq.heappop(queue)
        v = V[s]
        V[s] = max(sum(p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]) for a in range(env.action_space.n))
        errors.append(np.linalg.norm(V - V_star))
        iters.append(i + 1)
        if abs(v - V[s]) >= epsilon:
            for a in range(env.action_space.n):
                for p, s_, r, _ in env.P[s][a]:
                    heapq.heappush(queue, (-abs(V[s_] - V[s]), s_))
    return iters, errors


def policy_iteration(env, V_star, gamma=GAMMA, max_iterations=MAX_ITERATIONS):
    iterations = []
    errors_list = []
    for _ in range(5):
        policy = np.random.randn(env.observation_space.n)
        V = np.random.randn(env.observation_space.n)
        errors = []
        for i in range(max_iterations):
            old_policy = np.copy(policy)
            for s in range(env.observation_space.n):
                action_values = [sum(p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]) for a in
                                 range(env.action_space.n)]
                policy[s] = np.argmax(action_values)
            errors.append(np.linalg.norm(V - V_star))
            # print(f"Old policy = {old_policy}")
            # print(f"Current policy = {policy}")
            if np.array_equal(policy, old_policy):
                iterations.append(i + 1)
                break
        errors_list.append(errors)
    avg_iters = (np.mean(iterations))
    avg_errors = np.mean(errors_list, axis=0).tolist()
    return avg_iters, avg_errors


def plot_convergence(gs_iters, gs_errors, ps_iters, ps_errors, pi_iters, pi_errors, title, filename):
    plt.figure()
    plt.plot(gs_iters, gs_errors, label="Gauss-Seidel VI")
    plt.plot(ps_iters, ps_errors, label="Prioritized Sweeping VI")
    plt.plot(range(len(pi_errors)), pi_errors, label="Policy Iteration (Avg)")
    plt.xlabel("Number of iterations")
    plt.ylabel("||V - V*||_2")
    plt.legend()
    plt.title(title)
    plt.savefig(filename)
    plt.show()


def run_experiments(env):
    V_star = value_iteration(env)
    gs_iters, gs_errors = gauss_seidel_value_iteration(env, V_star)
    ps_iters, ps_errors = prioritized_sweeping_vi(env, V_star)
    pi_iters, pi_errors = policy_iteration(env, V_star)

    print(f"{env.spec.id}: Gauss-Seidel VI iterations: {len(gs_iters)}")
    print(f"{env.spec.id}: Prioritized Sweeping VI iterations: {len(ps_iters)}")
    print(f"{env.spec.id}: Policy Iteration avg iterations: {pi_iters}")

    plot_convergence(gs_iters, gs_errors, ps_iters, ps_errors, pi_iters, pi_errors,
                     f"Convergence for {env.spec.id}", f"../results/{env.spec.id}_convergence.png")


if __name__ == "__main__":
    taxi = gym.make("Taxi-v3")
    frozen_lake = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    run_experiments(taxi)
    run_experiments(frozen_lake)
