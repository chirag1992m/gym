import numpy as np
from six import StringIO
import sys

from gym import error
from gym import Env
from gym import spaces
from gym.utils import seeding


def make_tic_tac_toe_random_policy(np_random):
    def random_policy(state):
        possible_moves = TicTacToeEnv.get_possible_actions(state)
        # No moves left
        if len(possible_moves) == 0:
            return None
        a = np_random.randint(len(possible_moves))
        return possible_moves[a]
    return random_policy


class TicTacToeEnv(Env):
    """
    Tic-Tac-Toe Environment to play against a fixed opponent
    """
    metadata = {"render.modes": ["ansi", "human"]}
    NAUGHT = 0
    CROSS = 1

    def __init__(self, illegal_move_mode, board_size):
        """
        Args:
            illegal_move_mode: What to do when an agent makes an illegal move.
                Either 'raise' or 'lose'
            board_size: Size of the Tic-Tac-Toe board
        """
        assert (isinstance(board_size, int) and board_size > 1), \
            "Invalid board size {}".format(board_size)
        self.board_size = board_size

        assert illegal_move_mode in ['lose', 'raise'], \
            'Unsupported illegal move action: {}'.format(illegal_move_mode)
        self.illegal_move_mode = illegal_move_mode
        self.action_space = spaces.Discrete(self.board_size**2)
        self.observation_space = spaces.Box(0, 1, shape=(3, board_size, board_size))

        self.num_players = 2

        self.seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state = TicTacToeEnv.new_state(self.board_size)
        self.chance = TicTacToeEnv.CROSS
        self.done = [False] * self.num_players
        self.reward = [0.] * self.num_players

        return self.get_state()

    def switch_player(self):
        self.chance = 1 - self.chance

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        outfile.write('-' * (self.board_size * 4))
        outfile.write('\n')
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.state[2, x, y] == 1:
                    outfile.write('   ')
                else:
                    if self.state[0, x, y] == 1:
                        outfile.write(' O ')
                    else:
                        outfile.write(' X ')
                if y != self.board_size - 1:
                    outfile.write('|')
            outfile.write('\n')
            outfile.write('-' * (self.board_size * 4))
            outfile.write('\n')

        if mode != 'human':
            return outfile

    def get_state(self):
        return [self.state] * self.num_players

    def get_obs(self):
        return self.get_state(), self.reward, self.done, {}

    def _step(self, action):
        player_id, action = action
        assert self.chance == player_id
        # If already terminal, then don't do anything
        if all(self.done):
            self.reward = [0.] * self.num_players
            return self.get_obs()

        if not TicTacToeEnv.valid_move(self.state, action):
            if self.illegal_move_mode == 'raise':
                raise error.Error('Invalid move action: {}'.format(action))
            elif self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                self.done = [True] * self.num_players
                self.reward[player_id] = -1.
                self.reward[1 - player_id] = 1.0
                return self.get_obs()

        TicTacToeEnv.make_move(self.state, action, self.chance)
        self.check_and_update_internals()

        self.switch_player()

        return self.get_obs()

    def check_and_update_internals(self):
        """
        Updates internal states based on the current
        state of the board
        """
        winner = TicTacToeEnv.check_winner(self.state)
        if winner == 1:
            self.done = [True] * self.num_players
            self.reward[TicTacToeEnv.NAUGHT] = 1.0
            self.reward[TicTacToeEnv.CROSS] = -1.0
        elif winner == 2:
            self.done = [True] * self.num_players
            self.reward[TicTacToeEnv.NAUGHT] = -1.0
            self.reward[TicTacToeEnv.CROSS] = 1.0
        elif winner == 3:
            self.done = [True] * self.num_players
            self.reward[TicTacToeEnv.NAUGHT] = 0.
            self.reward[TicTacToeEnv.CROSS] = 0.

    @staticmethod
    def check_winner(state):
        """
        Gives the winner of the game.
        If Naughts, return 1
        If Crosses, return 2
        If Draw, return 3
        If None, return 0
        """
        naughts = state[TicTacToeEnv.NAUGHT]
        crosses = state[TicTacToeEnv.CROSS]

        board_size = state.shape[-1]

        if (np.any(np.sum(naughts, axis=0) == board_size) or
                np.any(np.sum(naughts, axis=1) == board_size)):
            return 1
        if (np.sum(np.diag(naughts)) == board_size or
                    np.sum(np.diag(np.flipud(naughts))) == board_size):
            return 1

        if (np.any(np.sum(crosses, axis=0) == board_size) or
                np.any(np.sum(crosses, axis=1) == board_size)):
            return 2
        if (np.sum(np.diag(crosses)) == board_size or
                    np.sum(np.diag(np.flipud(crosses))) == board_size):
            return 2

        # Checking for Draw
        if np.sum(np.logical_not(state[2, :])) == board_size ** 2:
            return 3

        return 0

    @staticmethod
    def new_state(board_size):
        state = np.zeros((3, board_size, board_size), dtype=np.int8)
        state[2, :] = 1
        return state

    @staticmethod
    def make_move(state, action, player_label):
        board_size = state.shape[-1]
        coordinate = TicTacToeEnv.action_to_coordinate(action, board_size)
        state[(2,) + coordinate] = 0
        state[(player_label,) + coordinate] = 1

    @staticmethod
    def valid_move(state, action):
        board_size = state.shape[-1]
        if 0 <= action < board_size ** 2:
            coordinate = TicTacToeEnv.action_to_coordinate(action, board_size)
            if state[(2,) + coordinate] == 1:
                return True
        return False

    @staticmethod
    def coordinate_to_action(coordinate, board_size):
        x, y = coordinate
        return x * board_size + y

    @staticmethod
    def action_to_coordinate(action, board_size):
        x = action // board_size
        y = action % board_size
        return x, y

    @staticmethod
    def get_possible_actions(state):
        free_x, free_y = np.where(state[2, :] == 1)
        return [TicTacToeEnv.coordinate_to_action((x, y), state.shape[0])
                for x, y in zip(free_x, free_y)]
