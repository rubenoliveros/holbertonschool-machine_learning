#!/usr/bin/env python3
"""Temporal Difference module"""
import numpy as np


def play_episode(env, policy, max_steps):
    """Plays a single episode
    Arguments:
        env {gym.Environment} -- Is the play environment
        policy {function} -- Is the policy function
        max_steps {int} -- Is the max steps per episode
    Returns:
        list(tuple) -- Contains the result for each step
    """
    state = env.reset()
    action = policy(state)
    state_action_reward = [(state, action, None)]

    for _ in range(max_steps):
        state, reward, done, _ = env.step(action)
        action = policy(state)
        state_action_reward.append((state, action, reward))
        if done:
            break

    return state_action_reward


def td_lambtha(
    env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99
):
    """Performs TD-lambtha backward prediction
    Arguments:
        env {gym.Environment} -- Is the play environment
        V {np.ndarray} -- Contains the value function
        policy {function} -- Is the policy function
        lambtha {float} -- Is the decay parameter
    Keyword Arguments:
        episodes {int} -- Is the number of episodes (default: {5000})
        max_steps {int} -- Is the number of max steps (default: {100})
        alpha {float} -- Is the learning rate (default: {0.1})
        gamma {float} -- Is the discount rate (default: {0.99})
    Returns:
        np.ndarray -- The updated value function
    """
    for _ in range(episodes):
        ET = 0
        state_action_reward = play_episode(env, policy, max_steps)
        T = len(state_action_reward) - 1

        for t in range(T):
            state, _, _ = state_action_reward[t]
            state_t_1, _, reward_t_1 = state_action_reward[t + 1]

            ET *= lambtha * gamma
            ET += 1

            delta = reward_t_1 + gamma * V[state_t_1] - V[state]
            V[state] += alpha * delta * ET
