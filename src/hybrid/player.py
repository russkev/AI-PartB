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
from strategy.mcts_duct import Node, monte_carlo_tree_search, simple_reduction
from strategy.evaluation import greedy_choose
import cProfile
from strategy.minimax import minimax_paranoid_reduction
from strategy.book import book_first_four_moves

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
        self.root.pruning_is_aggressive = True
        self.start_time = self.end_time = self.time_consumed = 0

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """

        self.start_timer()


        if self.root.turn < 4:
            result = book_first_four_moves(self.root)
        elif self.root.branching < 360:
            print("USING MCTS")
            result = monte_carlo_tree_search(
                self.root, 
                playout_amount = 3, 
                node_cutoff = 3,
                outer_cutoff = 3,
                num_iterations = 900, 
                turn_time = 0.85, 
                exploration_constant = 0.8,
                use_slow_culling = False,
                verbosity = 0,
                use_prior = True,
                num_priors = 10,
            )
        else:
            print("USING MINIMAX")
            result = minimax_paranoid_reduction(self.root)

        return result

    def update(self, opponent_action, player_action):
        """0
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """

        self.root = self.root.make_updated_node(player_action,opponent_action)

    def start_timer(self):
        self.start_time = time()


    def end_timer(self):
        self.end_time = time()
        self.time_consumed += self.end_time - self.start_time
