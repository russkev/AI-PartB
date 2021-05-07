"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

import csv
import random
import time
from state.game_state import GameState
from strategy.minimax import minimax_with_ml
from strategy.evaluation_features import EvaluationFeatures


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
        self.evaluation_features = EvaluationFeatures()
        self.out_file = 'game_logs/' + str(int(time.time())) + '.csv'
        self.explore_rate = 0.1

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        
        if random.random() < self.explore_rate:
            transitions = self.game_state.next_transitions_for_side(True)
            return transitions[random.randrange(len(transitions))]

        return minimax_with_ml(self.game_state)
        
    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        self.game_state.update(player_action, opponent_action)

        self.evaluation_features.calculate_features(self.game_state)

        with open(self.out_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(self.evaluation_features.to_vector())




