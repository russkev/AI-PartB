"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from state.game_state import GameState
from strategy.minimax import minimax_paranoid_reduction
import numpy as np
from numpy.random import choice
from strategy.evaluation import evaluate_state
from strategy.nash import solve_game

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

        return minimax_simultaneous(self.game_state)
        
    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        self.game_state = self.game_state.update(opponent_action, player_action)

    

def minimax_simultaneous(game_state: GameState):
    fr_transitions = game_state.next_friend_transitions()

    strategy, _ = evaluate_all_moves(game_state, fr_transitions, 2)

    i = choice(range(len(fr_transitions)), p=strategy)

    return fr_transitions[i]


    # fr_transitions = game_state.next_enemy_transitions()

    # pass


def evaluate_all_moves(game_state: GameState, fr_transitions, depth):
    weights = [1, 100, -10, -5, -1, -0.5]
    range = 0.3

    if depth == 0:
        score, _ = evaluate_state(game_state, weights)
        return None, score


    goal_score = game_state.goal_reward()
    if goal_score is not None:
        return None, goal_score * 200


    en_transitions = game_state.next_enemy_transitions()

    value_matrix = []

    best_score = -1000

    for fr_transition in fr_transitions:
        row = []
        for en_transition in en_transitions:
            state_ij = game_state.update(en_transition, fr_transition)
            score_ij, _ = evaluate_state(state_ij, weights)
            if score_ij > best_score:
                best_score = score_ij
            if score_ij > best_score - range:
                fr_transitions_ij = state_ij.next_friend_transitions()
                _, score_ij = evaluate_all_moves(state_ij, fr_transitions_ij, depth-1)
            row.append(score_ij)
        value_matrix.append(row)
    
    return solve_game(value_matrix)

    # solve, value = solve_game(value_matrix)

    # i = choice(range(len(fr_transitions)), p=solve)

    # return fr_transitions[i]






# def smab(game_state: GameState, lower_bound, upper_bound):
#     goal_reward = game_state.goal_reward
#     if goal_reward != None:
#         return goal_reward
    

#     fr_transitions = game_state.next_friend_transitions()
#     en_transitions = game_state.next_enemy_transitions()
#     pessimistic = np.zeros(shape=(len(fr_transitions), len(en_transitions)))
#     optimistic = np.zeros(shape=(len(fr_transitions), len(en_transitions)))

#     for i, fr_transition in enumerate(fr_transitions):
#         for j, en_transition in enumerate(en_transitions):
#             if not_dominated():
#                 lower_bound_ij = non_dominated_actions()
#                 upper_bound_ij = non_dominated_actions()
#                 state_ij = game_state.update(en_transition, fr_transition)
#                 if lower_bound_ij >= upper_bound_ij:
#                     value_ij = smab(state_ij, lower_bound_ij, lower_bound_ij + epsilon)
#                     if value_ij <= lower_bound_ij:
#                         dominated = fr_transition
#                     else:
#                         dominated = en_transition
#                 else:
#                     value_ij = smab(state_ij, lower_bound_ij, upper_bound_ij)
#                     if value_ij <= lower_bound_ij:
#                         dominated = fr_transition
#                     elif value_ij >= upper_bound:
#                         dominated = en_transition
#                     else:
#                         pessimistic[i][j] = value_ij
#                         optimistic[i][j] = value_ij
#     return nash(pessimistic)



