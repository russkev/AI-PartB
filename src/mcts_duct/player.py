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
from strategy.mcts_duct import Node, monte_carlo_tree_search, test
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
        game_state = GameState()
        self.root = Node(game_state)
        if player == "lower":
            self.root.is_upper = False

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        # test()
        random_turns = 20
        if self.root.turn < random_turns:
            return greedy_choose(self.root)
        else:
            return monte_carlo_tree_search(self.root, 10000)


    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """

        self.root = self.root.update_node(opponent_action, player_action)
