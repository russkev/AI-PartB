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
# from strategy.evaluation import eval_function
# from strategy import evaluation
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
            eval_score, scores = evaluation_score(new_state, enemy_transitions)
            heappush(queue, (-1 * eval_score, tuple(scores), friend_transition))

        # queue = opponent_distance_scores(self.game_state)
        (best_score, best_scores, best_move) = heappop(queue)
        possible_moves = [(best_score, best_scores, best_move)]

        # Add all moves with the same best score to a list
        for (curr_score, *rest) in queue:
            if curr_score != best_score:
                break
            else:
                possible_moves.append((curr_score, *rest))


        # Randomly pick from that list
        return possible_moves[np.random.choice(len(possible_moves), 1)[0]][2]

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

def evaluation_score(game_state: GameState, enemy_transitions):


    dist_to_killable_score_friend = eval.distance_to_killable_score(game_state, is_friend=True)
    dist_to_killable_score_enemy = eval.distance_to_killable_score(game_state, is_friend=False)
    dist_to_killable_score_diff = dist_to_killable_score_friend - dist_to_killable_score_enemy

    num_friend_useless, num_enemy_useless = eval.num_useless(game_state)
    num_useless_diff = num_friend_useless - num_enemy_useless

    num_killed_diff = eval.num_opponents_killed_difference(game_state)
    pieces_in_throw_range_diff = eval.pieces_in_throw_range_difference(game_state)

    friend_move_to_pieces = game_state.moves_to_pieces(game_state.next_friend_moves(), is_friend=True)
    enemy_move_to_pieces = game_state.moves_to_pieces(game_state.next_enemy_moves(), is_friend=False)

    pieces_in_move_range_diff = eval.num_can_be_move_killed_difference(game_state, friend_move_to_pieces, enemy_move_to_pieces)

    scores = [
        dist_to_killable_score_diff,
        num_killed_diff,
        num_useless_diff,
        pieces_in_throw_range_diff,
        pieces_in_move_range_diff
    ]

    weights = [
        1,
        50,
        -10,
        -5,
        -1
    ]

    final_scores = np.multiply(scores, weights)

    return np.dot(scores, weights), final_scores
        
