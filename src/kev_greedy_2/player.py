"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from random import randrange
from heapq import heappush, heappop

from state.game_state import GameState
from state.token import defeats
from state.location import distance
from strategy.greedy_util import opponent_distance_scores
from strategy.evaluation import eval_function


class Player:

    def __init__(self, player):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "upper" (if the instance will
        play as Upper), or the string "lower" (if the instance will play
        as Lower).
        """
        self.game_state = GameState()
        self.game_state.is_upper = player == "upper"

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        friend_transitions = self.game_state.next_friend_transitions()
        enemy_transitions = self.game_state.next_enemy_transitions()
        num_enemy_transitions = len(enemy_transitions)

        queue = []
        for friend_transition in friend_transitions:
            # enemy_transition = enemy_transitions[randrange(num_enemy_transitions)]
            new_state = self.game_state.update(friend_transition=friend_transition)
            eval_score, scores = eval_function(new_state, enemy_transitions)
            heappush(queue, (-1 * eval_score, tuple(scores), friend_transition))

        # queue = opponent_distance_scores(self.game_state)
        (best_score, best_scores, best_move) = heappop(queue)
        return best_move

    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        self.game_state = self.game_state.update(
            opponent_action, player_action)
