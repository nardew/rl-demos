import itertools
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, t):
        t = self.net(t)
        return t

    def act(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        q_values = self(state_t.unsqueeze(0))

        max_q = torch.argmax(q_values, dim=1)[0]
        action = max_q.item()

        return action


def decay(eps_start, eps_end, eps_decayrate, current_step):
    return eps_end + (eps_start - eps_end) * np.exp(-eps_decayrate * current_step)

env = gym.make("Acrobot-v1", render_mode="rgb_array")


steps_all = 0
batch_size = 32
# batch_size = 2
gamma = 0.99

online_net = Network()
target_net = Network()
target_net.load_state_dict(online_net.state_dict())

epsilon_start = 1
epsilon_end = 0.001
epsilon_decayrate = 0.00001

episode_durations = []

optimizer = optim.Adam(online_net.parameters(), lr=5e-4)

replayMemory = []
state, _ = env.reset()


for _ in (range(1000)):
    action = env.action_space.sample()
    s1, reward, done, _, _ = env.step(action)
    experience = (state, action, reward, done, s1)
    replayMemory.append(experience)
    state = s1

    if done:
        env.reset()

try:
    perf = []
    for t in range(100000):
        state, _ = env.reset()

        for step in itertools.count():
            steps_all += 1
            epsilon = decay(epsilon_start, epsilon_end, epsilon_decayrate, steps_all)

            if random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                action = online_net.act(state)

            s1, reward, done, _, _ = env.step(action)
            experience = (state, action, reward, done, s1)
            replayMemory.append(experience)
            state = s1

            if steps_all % 10 == 0:
                experiences = random.sample(replayMemory, batch_size)
                # experiences = replayMemory[-batch_size:]
                states = np.asarray([e[0] for e in experiences], dtype=np.float32)
                actions = np.asarray([e[1] for e in experiences], dtype=np.int64)  # Actions are integers
                rewards = np.asarray([e[2] for e in experiences], dtype=np.float32)
                dones = np.asarray([e[3] for e in experiences], dtype=np.float32)
                new_states = np.asarray([e[4] for e in experiences], dtype=np.float32)

                states_t = torch.as_tensor(states, dtype=torch.float32)
                actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
                rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
                dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
                new_states_t = torch.as_tensor(new_states, dtype=torch.float32)

                q_values = online_net(states_t)
                action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

                target_q_output = target_net(new_states_t)
                # target_q_output = online_net(new_states_t)
                target_q_values = target_q_output.max(dim=1, keepdim=True)[0]
                optimal_q_values = rewards_t + gamma * target_q_values * (1 - dones_t)

                loss = nn.functional.smooth_l1_loss(action_q_values, optimal_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_all % 200 == 0:
                target_net.load_state_dict(online_net.state_dict())

            if t > 5000:
                env = gym.wrappers.RecordVideo(env, fps=10, video_folder="./", episode_trigger=lambda x: True)
                state, _ = env.reset()
                exec_step = 0
                while True:
                    exec_step += 1
                    action = online_net.act(state)
                    state, _, done, _, info = env.step(action)

                    if done and exec_step > 1000:
                        print(f"Step {exec_step}")

                        # exec_step = 0
                        # state, _ = env.reset()
                        exit(0)

            if done:
                print(f"Episode {t}, steps {step}, epsilon {epsilon}")
                perf.append(step)
                break
finally:
    env.close()
