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

    def __init__(self, player_label, opponent, illegal_move_mode, board_size):
        """
        Args:
            player_label: Label for the agent. Either 'RED' or 'BLUE'
            opponent: Opponent policy
            illegal_move_mode: What to do when an agent makes an illegal move.
                Either 'raise' or 'lose'
            board_size: Size of the Connect-Four board
        """
        assert (isinstance(board_size, int) and board_size > 1), \
            "Invalid board size {}".format(board_size)
        self.board_size = board_size
        self.board_shape = (3, board_size, board_size)

        colormap = {
            'red': ConnectFourEnv.RED,
            'blue': ConnectFourEnv.BLUE
        }
        try:
            self.player_label = colormap[player_label]
        except KeyError:
            raise error.Error(
                "player_color must be 'red' or 'blue', not {}".format(player_label))

        self.opponent = opponent

        assert illegal_move_mode in ['lose', 'raise'], \
            'Unsupported illegal move action: {}'.format(illegal_move_mode)
        self.illegal_move_mode = illegal_move_mode
        self.action_space = spaces.Discrete(self.board_size)
        self.observation_space = spaces.Box(0, 1, shape=self.board_shape)

        self.seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Update the random policy if needed
        if isinstance(self.opponent, str):
            if self.opponent == 'random':
                self.opponent_policy = make_connect_four_random_policy(self.np_random)
            else:
                raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))
        else:
            self.opponent_policy = self.opponent

        return [seed]

    def _reset(self):
        self.state = np.zeros(self.board_shape, dtype=np.int8)
        self.state[2, :] = 1
        self.height = np.zeros(self.board_size, dtype=np.int8)
        self.chance = ConnectFourEnv.BLUE
        self.done = False

        if self.player_label != self.chance:
            action = self.opponent_policy(self.state)
            self.make_move(action, self.chance)
            self.switch_player()

        return self.state

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
                        outfile.write(' R ')
                    else:
                        outfile.write(' B ')
                if y != self.board_size - 1:
                    outfile.write('|')
            outfile.write('\n')
            outfile.write('-' * (self.board_size * 4))
            outfile.write('\n')

        if mode != 'human':
            return outfile

    def _step(self, action):
        assert self.chance == self.player_label
        # If already terminal, then don't do anything
        if self.done:
            return self.state, 0., True, {'state': self.state}

        if not self.valid_move(action):
            if self.illegal_move_mode == 'raise':
                raise error.Error('Invalid move action: {}'.format(action))
            elif self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                self.done = True
                return self.state, -1., True, {'state': self.state}

        self.make_move(action, self.chance)
        winner = self.game_finished()
        if winner != -1:
            reward = 1 if winner == self.player_label else -1
            self.done = True
            return self.state, reward, self.done, {'state': self.state}

        self.switch_player()
        # Opponent play
        a = self.opponent_policy(self.state)
        if a is not None:
            self.make_move(a, self.chance)
        self.switch_player()

        winner = self.game_finished()
        reward = 0
        if winner != -1:
            reward = 1 if winner == self.player_label else -1
            self.done = True
        return self.state, reward, self.done, {'state': self.state}

    def make_move(self, action, player_label):
        ht = self.height[action]
        self.state[player_label][action][ht] = 1
        self.state[2][action][ht] = 0
        self.height[action] += 1

    def check_win(self, player_label):
        max_consecutive = 0
        for row in range(self.board_size):
            cur = 0
            for col in range(self.board_size):
                if self.state[player_label][row][col]:
                    cur += 1
                    max_consecutive = max(max_consecutive, cur)
                else:
                    cur = 0
            if max_consecutive >= 4:
                return True
        for col in range(self.board_size):
            cur = 0
            for row in range(self.board_size):
                if self.state[player_label][row][col]:
                    cur += 1
                    max_consecutive = max(max_consecutive, cur)
                else:
                    cur = 0
            if max_consecutive >= 4:
                return True
        for row in range(self.board_size):
            for col in range(self.board_size):
                if row <= self.board_size - 4 and col <= self.board_size - 4:
                    cur = 0
                    for delta in range(4):
                        if self.state[player_label][row+delta][col+delta]:
                            cur += 1
                    if cur >= 4:
                        return True
                if row <= self.board_size - 4 and col >= 3:
                    cur = 0
                    for delta in range(4):
                        if self.state[player_label][row+delta][col-delta]:
                            cur += 1
                    if cur >= 4:
                        return True
        return False

    def game_finished(self):
        """
        :return: -1, if nobody has won the game.
            BLUE if BLUEes has won
            RED if REDs has won
        """
        if self.check_win(ConnectFourEnv.RED):
            return ConnectFourEnv.RED
        if self.check_win(ConnectFourEnv.BLUE):
            return ConnectFourEnv.BLUE

        if np.sum(np.logical_not(self.state[2, :])) == self.board_size**2:
            self.done = True

        return -1

    def valid_move(self, action):
        if 0 <= action < self.board_size:
            if self.height[action] < self.board_size:
                return True
        return False

    @staticmethod
    def get_possible_actions(state):
        free_x, _ = np.where(state[2, :] == 1)
        free_actions = list(set(free_x))
        return free_actions
