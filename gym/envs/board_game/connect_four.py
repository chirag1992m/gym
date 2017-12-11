import numpy as np
from six import StringIO
import sys

from gym import error
from gym import Env
from gym import spaces
from gym.utils import seeding


def make_connect_four_random_policy(np_random):
    def random_policy(state):
        possible_moves = ConnectFourEnv.get_possible_actions(state)
        # No moves left
        if len(possible_moves) == 0:
            return None
        a = np_random.randint(len(possible_moves))
        return possible_moves[a]
    return random_policy


class ConnectFourEnv(Env):
    """
    Tic-Tac-Toe Environment to play against a fixed opponent
    """
    metadata = {"render.modes": ["ansi", "human"]}
    RED = 0
    BLUE = 1

    def __init__(self, illegal_move_mode, board_size):
        """
        Args:
            illegal_move_mode: What to do when an agent makes an illegal move.
                Either 'raise' or 'lose'
            board_size: Size of the Connect-Four board
        """
        assert (isinstance(board_size, int) and board_size > 1), \
            "Invalid board size {}".format(board_size)
        self.board_size = board_size

        assert illegal_move_mode in ['lose', 'raise'], \
            'Unsupported illegal move action: {}'.format(illegal_move_mode)
        self.illegal_move_mode = illegal_move_mode
        self.action_space = spaces.Discrete(self.board_size)
        self.observation_space = spaces.Box(0, 1, shape=(3, board_size, board_size))

        self.num_players = 2

        self.seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self):
        return [self.state] * self.num_players

    def _reset(self):
        self.state = np.zeros((3, self.board_size, self.board_size), dtype=np.int8)
        self.state[2, :] = 1
        self.height = np.zeros(self.board_size, dtype=np.int8)
        self.chance = ConnectFourEnv.BLUE
        self.done = [False] * self.num_players
        self.reward = [0.] * self.num_players

        return self.get_state()

    def get_obs(self):
        return self.get_state(), self.reward, self.done, {'height': self.height}

    def switch_player(self):
        self.chance = 1 - self.chance

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        ConnectFourEnv.render_to_file(self.state, outfile)

        if mode != 'human':
            return outfile

    @staticmethod
    def render_to_file(state, outfile):
        board_size = state.shape[-1]
        outfile.write('-' * (board_size * 4))
        outfile.write('\n')
        for x in range(board_size):
            for y in range(board_size):
                if state[2, x, y] == 1:
                    outfile.write('   ')
                else:
                    if state[0, x, y] == 1:
                        outfile.write(' R ')
                    else:
                        outfile.write(' B ')
                if y != board_size - 1:
                    outfile.write('|')
            outfile.write('\n')
            outfile.write('-' * (board_size * 4))
            outfile.write('\n')

    def _step(self, action):
        player_id, action = action
        assert self.chance == player_id
        # If already terminal, then don't do anything
        if all(self.done):
            self.reward = [0.] * self.num_players
            return self.get_obs()

        if not ConnectFourEnv.valid_move(self.height, self.board_size, action):
            if self.illegal_move_mode == 'raise':
                raise error.Error('Invalid move action: {}'.format(action))
            elif self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                self.done = [True] * self.num_players
                self.reward[player_id] = -1.
                self.reward[1 - player_id] = 1.0
                return self.get_obs()

        ConnectFourEnv.make_move(self.state, self.height, action, self.chance)
        self.check_and_update_internals()

        self.switch_player()

        return self.get_obs()

    def check_and_update_internals(self):
        """
        Updates internal states based on the current
        state of the board
        """
        winner = ConnectFourEnv.get_winner(self.state)
        if winner == 1:
            self.done = [True] * self.num_players
            self.reward[ConnectFourEnv.RED] = 1.0
            self.reward[ConnectFourEnv.BLUE] = -1.0
        elif winner == 2:
            self.done = [True] * self.num_players
            self.reward[ConnectFourEnv.RED] = -1.0
            self.reward[ConnectFourEnv.BLUE] = 1.0
        elif winner == 3:
            self.done = [True] * self.num_players
            self.reward[ConnectFourEnv.RED] = 0.
            self.reward[ConnectFourEnv.BLUE] = 0.

    @staticmethod
    def make_move(state, height, action, player_label):
        ht = height[action]
        state[player_label][action][ht] = 1
        state[2][action][ht] = 0
        height[action] += 1

    @staticmethod
    def get_winner(state):
        """
        Gives the winner of the game.
        If RED, return 1
        If BLUE, return 2
        If Draw, return 3
        If None, return 0
        """
        if ConnectFourEnv.check_win(state, ConnectFourEnv.RED):
            return 1
        if ConnectFourEnv.check_win(state, ConnectFourEnv.BLUE):
            return 2
        if np.sum(np.logical_not(state[2, :])) == state.shape[-1] ** 2:
            return 3
        return 0

    @staticmethod
    def check_win(state, player_label):
        max_consecutive = 0
        board_size = state.shape[-1]
        for row in range(board_size):
            cur = 0
            for col in range(board_size):
                if state[player_label][row][col]:
                    cur += 1
                    max_consecutive = max(max_consecutive, cur)
                else:
                    cur = 0
            if max_consecutive >= 4:
                return True
        for col in range(board_size):
            cur = 0
            for row in range(board_size):
                if state[player_label][row][col]:
                    cur += 1
                    max_consecutive = max(max_consecutive, cur)
                else:
                    cur = 0
            if max_consecutive >= 4:
                return True
        for row in range(board_size):
            for col in range(board_size):
                if row <= board_size - 4 and col <= board_size - 4:
                    cur = 0
                    for delta in range(4):
                        if state[player_label][row + delta][col + delta]:
                            cur += 1
                    if cur >= 4:
                        return True
                if row <= board_size - 4 and col >= 3:
                    cur = 0
                    for delta in range(4):
                        if state[player_label][row + delta][col - delta]:
                            cur += 1
                    if cur >= 4:
                        return True
        return False

    @staticmethod
    def valid_move(height, board_size, action):
        if 0 <= action < board_size:
            if height[action] < board_size:
                return True
        return False

    @staticmethod
    def get_possible_actions(state):
        free_x, _ = np.where(state[2, :] == 1)
        free_actions = list(set(free_x))
        return free_actions
