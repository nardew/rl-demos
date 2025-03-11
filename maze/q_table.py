from typing import List
from environment import Action, Environment, Position, ENV_ROWS, ENV_COLS, convert_action_to_position, is_valid_position

class QTableActionValue:
    def __init__(self, action: Action, value: float):
        self.action = action
        self.value = value


class QTableCell:
    def __init__(self):
        self.action_values: List[QTableActionValue] = [
            QTableActionValue(Action.UP, 0.0),
            QTableActionValue(Action.DOWN, 0.0),
            QTableActionValue(Action.LEFT, 0.0),
            QTableActionValue(Action.RIGHT, 0.0),
        ]

    def get_action_value(self, action: Action) -> QTableActionValue:
        return next((action_value for action_value in self.action_values if action_value.action == action))


class QTable:
    def __init__(self, env: Environment):
        self.env = env

        self.q_table: List[List[QTableCell]] = []
        for row_i in range(ENV_ROWS):
            row = []
            for col_i in range(ENV_COLS):
                row.append(QTableCell())
            self.q_table.append(row)

    def get_cell(self, position: Position) -> QTableCell:
        return self.q_table[position.row][position.col]

    def get_action_value(self, position: Position, action: Action) -> QTableActionValue:
        action_values = self.q_table[position.row][position.col].action_values
        return next((action_value for action_value in action_values if action_value.action == action))

    def get_best_action(self, position: Position) -> QTableActionValue:
        available_actions = []
        for action_value in self.q_table[position.row][position.col].action_values:
            new_position = convert_action_to_position(position, action_value.action)
            if is_valid_position(self.env, new_position):
                available_actions.append(action_value)

        return max(available_actions, key=lambda x: x.value)
