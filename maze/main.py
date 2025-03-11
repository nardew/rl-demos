import random

import numpy as np

from environment import Position, Environment, convert_action_to_position
from video import VideoRecorder
from q_table import QTable


epsilon_start = 1
epsilon_end = 0.001
epsilon_decay_rate = 0.0025


def epsilon_decay(eps_start, eps_end, eps_decay_rate, current_step):
    return eps_end + (eps_start - eps_end) * np.exp(-eps_decay_rate * current_step)


env = Environment()
q_table = QTable(env)
lambda_learning_rate = 0.95
gama_discount_rate = 0.9

step = 0
epsilon = epsilon_start
video_recorder = VideoRecorder()

for episode in range(200):
    print(f"Episode: {episode}")
    position = Position(1, 0)
    video_recorder.add_frame(position, q_table,
                             episode,
                             step,
                             epsilon,
                             None,
                             None,
                             None,
                             None,
                             lambda_learning_rate,
                             gama_discount_rate,
                             None,
                             None,
                             None)

    while True:
        step += 1
        epsilon = epsilon_decay(epsilon_start, epsilon_end, epsilon_decay_rate, step)
        random_number = random.random()
        if random_number > epsilon:
            # use Q table
            best_action_value = q_table.get_best_action(position)
            new_action = best_action_value.action
        else:
            # random action
            available_actions = env.get_available_actions(position)
            new_action = random.choice(available_actions)

        new_position = convert_action_to_position(position, new_action)
        current_action_value = q_table.get_action_value(position, new_action)
        new_best_action_value = q_table.get_best_action(new_position)
        reward = env.rewards[new_position.row][new_position.col]

        orig_current_action_value = current_action_value.value
        current_action_value.value = current_action_value.value + \
            lambda_learning_rate * (reward + gama_discount_rate * new_best_action_value.value - current_action_value.value)

        video_recorder.add_frame(new_position, q_table,
                                 episode,
                                 step,
                                 epsilon,
                                 position,
                                 new_action,
                                 random_number <= epsilon,
                                 reward,
                                 lambda_learning_rate,
                                 gama_discount_rate,
                                 orig_current_action_value,
                                 new_best_action_value.value,
                                 current_action_value.value)

        position = new_position

        if env.is_terminal(new_position):
            break

video_recorder.close()
