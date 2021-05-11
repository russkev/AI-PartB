"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

import random
from state.game_state import GameState
from strategy.minimax import minimax_paranoid_reduction
from strategy.evaluation import greedy_choose
from strategy.book import book_first_four_moves
from time import time


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
        self.game_state.pruning_is_aggressive = True
        self.start_time = self.end_time = self.time_consumed = 0


    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        
        self.start_timer()


        if self.root.turn < 4:
            result = book_first_four_moves(self.root)
        elif self.time_consumed < 59:
            result = minimax_paranoid_reduction(self.game_state)
        else:
            result = greedy_choose(self.game_state)

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
        self.game_state.update(player_action, opponent_action)

    
    def start_timer(self):
        self.start_time = time()

    def end_timer(self):
        self.end_time = time()
        self.time_consumed += self.end_time - self.start_time
