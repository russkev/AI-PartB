"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from time import time
from state.game_state import GameState
from strategy.book import book_first_four_moves
import numpy as np
from strategy.mcts_duct import Node, monte_carlo_tree_search
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
        else:
            result = monte_carlo_tree_search(
                self.root, 
                playout_amount = 3, 
                node_cutoff = 6,
                outer_cutoff = 6,
                num_iterations = 9000, 
                turn_time = 1, 
                exploration_constant = 0.8,
                use_slow_culling = False,
                verbosity = 1,
                use_prior = True,
                num_priors = 4,
                use_fast_prune_eval=False,
                use_fast_rollout_eval=False,
            )

        self.end_timer()

        return result

    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """

        self.root = self.root.make_updated_node(player_action,opponent_action)
        asda=4

    def start_timer(self):
        self.start_time = time()


    def end_timer(self):
        self.end_time = time()
        self.time_consumed += self.end_time - self.start_time
