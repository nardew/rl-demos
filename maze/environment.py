import enum
from typing import List


ENV_ROWS = 4
ENV_COLS = 5


class Position:
    def __init__(self, row: int, col: int):
        self.row: int = row
        self.col: int = col


class Action(enum.Enum):
    UP = enum.auto()
    DOWN = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()


class Environment:
    def __init__(self):
        self.grid = [
            [-1,  0,  0,  0,  0],
            [ 0,  0,  0,  1,  0],
            [-1,  1,  0,  0,  1],
            [-1,  0,  0, -1, -1],
        ]
        self.rewards = [
            [ 0,   -1, -1,   -1,  -1],
            [-1,   -1, -1,  -10,  -1],
            [ 0,  -10, -1,   -3,  10],
            [ 0,   -1, -1,    0,   0],
        ]

    def is_terminal(self, position: Position) -> bool:
        return self.grid[position.row][position.col] == 1

    def get_available_actions(self, position: Position) -> List[Action]:
        up = Position(position.row - 1, position.col)
        down = Position(position.row + 1, position.col)
        left = Position(position.row, position.col - 1)
        right = Position(position.row, position.col + 1)

        available_positions = []
        if is_valid_position(self, up):
            available_positions.append(Action.UP)
        if is_valid_position(self, down):
            available_positions.append(Action.DOWN)
        if is_valid_position(self, right):
            available_positions.append(Action.RIGHT)
        if is_valid_position(self, left):
            available_positions.append(Action.LEFT)

        return available_positions


def is_valid_position(env: Environment, position: Position) -> bool:
    if position.row < 0 or position.row >= ENV_ROWS or position.col < 0 or position.col >= ENV_COLS:
        return False

    if env.grid[position.row][position.col] == -1:
        return False

    return True


def convert_action_to_position(position: Position, action: Action) -> Position:
    row_delta = 0
    col_delta = 0

    if action == Action.UP:
        row_delta = -1
    if action == Action.DOWN:
        row_delta = 1
    if action == Action.LEFT:
        col_delta = -1
    if action == Action.RIGHT:
        col_delta = 1

    return Position(position.row + row_delta, position.col + col_delta)
