#!/usr/bin/env python3
"""
import numpy librairy
import gym
"""
import numpy as np
import gym


def generate_episode(env, policy, max_steps):
    """
    function generate episode
    """
    v = env.reset()
    a = []
    for i in range(max_steps):
        action = policy(v)
        next_state, reward, done, info = env.step(action)
        a.append((v, reward))
        v = next_state
        if done:
            break
    return a


def monte_carlo(env,
                V,
                policy,
                episodes=5000,
                max_steps=100,
                alpha=0.1,
                gamma=0.99):
    """
    monte carlo function
    """
    r = set()
    for i in range(1, episodes+1):
        a = generate_episode(env, policy, max_steps)
        b, rewards = zip(*a)
        d = np.array([gamma ** i for i in range(len(rewards) + 1)])
        for idx in range(len(a[0])-1, -1, -1):
            G = sum(rewards[idx:] * d[:-(idx+1)])
            if not a[idx][0] in r:
                V[a[idx][0]] = V[a[idx][0]] + \
                    alpha * (G - V[a[idx][0]])
            r.add(a[idx][0])
    return V
