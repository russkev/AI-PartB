"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from state.game_state import GameState
import strategy.evaluation as eval
import numpy as np


class Player:

    def __init__(self, player):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "upper" (if the instance will
        play as Upper), or the string "lower" (if the instance will play
        as Lower).
        """
        self.game_state = GameState(is_upper=(player == "upper"))

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.kev
        """
        return eval.greedy_choose(self.game_state, weights=[
            10,     # dist_to_killable_score_diff
            200,    # num_killed_diff
            -15,    # num_useless_diff
            -30,    # pieces_on_board_diff
            -25,    # pieces_in_throw_range_diff
            -25,    # pieces_in_move_range_diff
            -3,     # distance_from_safeline_diff
            500,    # invincible_diff
        ])

    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        self.game_state.update(player_action, opponent_action)
        
