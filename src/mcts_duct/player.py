"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from random import randrange
from time import time
from state.game_state import GameState
from strategy.rando_util import biased_random_move
import numpy as np
from strategy.mcts_duct import Node, monte_carlo_tree_search, test_2, test_3, simple_reduction
from strategy.evaluation import greedy_choose

class Player:

    def __init__(self, player):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "upper" (if the instance will
        play as Upper), or the string "lower" (if the instance will play
        as Lower).
        """
        self.root = Node(GameState(is_upper=(player == "upper")))

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        greedy_turns = 14
        if self.root.turn < greedy_turns:
            return greedy_choose(self.root)
        else:
            return monte_carlo_tree_search(self.root, num_iterations=300, playout_amount=6, node_cutoff=5)
            # return simple_reduction(self.root)
            # test_3()

    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """

        self.root = self.root.make_updated_node(player_action,opponent_action)

