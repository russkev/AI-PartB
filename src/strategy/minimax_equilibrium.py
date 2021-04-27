"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from state.game_state import GameState
import numpy as np
from numpy.random import choice
from strategy.evaluation import evaluate_state
from strategy.nash import solve_game
from strategy.evaluation import goal_reward

def minimax_equilibrium(game_state: GameState, depth=1, cutoff_range=0.3):
    """
    Minimax simultaneous move solution that uses linear programming to evaluate all possible 
    combinations of move and make a choice based on a mixed strategy

    """

    # Precalculation
    fr_transitions = game_state.next_friend_transitions()

    if depth == 1:
        # Use the simpler version of the algorithm
        strategy, _ = __evaluate_equilibrium(game_state, fr_transitions)
    else:
        # Use the more complicated version that evaluates to a certain depth in the tree
        strategy, _ = __evaluate_equilibrium_recursive(
            game_state, fr_transitions, depth, cutoff_range)

    # choose an index based on the probabilities in the mixed strategy
    i = choice(range(len(fr_transitions)), p=strategy)

    return fr_transitions[i]


def __evaluate_equilibrium(game_state: GameState, fr_transitions):
    """
    Construct a matrix of scores with friend being the rows and enemy the columns.

    When complete, run the linear programming algorithm to calculate the mixed strategy and the
    max score.
    """

    # Check for terminal state
    goal_score = goal_reward(game_state)
    if goal_score is not None:
        return None, goal_score * 1000
    
    value_matrix = []
    en_transitions = game_state.next_enemy_transitions()
    for fr_transition in fr_transitions:
        value_matrix.append(__row_scores(game_state, fr_transition, en_transitions))
    return solve_game(value_matrix)

def __evaluate_equilibrium_recursive(game_state: GameState, fr_transitions, depth, cutoff_range):
    """
    Construct a matrix of scores with friend being the rows and enemy the columns.

    If a score is above the best_score - cutoff range, recursively call the evaluation down to the
    maximum depth.

    Once scores have been calculated, run the linear programming algorithm to determine a mixed
    strategy.
    """

    # Base case
    if depth == 0:
        score, _ = evaluate_state(game_state)
        return None, score

    # Check for terminal state
    goal_score = goal_reward(game_state)
    if goal_score is not None:
        return None, goal_score * 1000
    value_matrix = []

    # Main loop
    en_transitions = game_state.next_enemy_transitions()
    best_score = float("-inf")
    for fr_transition in fr_transitions:
        row, best_score = __row_scores_recursive(
            game_state, fr_transition, en_transitions, depth, cutoff_range, best_score)
        value_matrix.append(row)
    
    # Linear programming step
    return solve_game(value_matrix)


def __row_scores(game_state: GameState, fr_transition, en_transitions):
    """
    Evaluate the scores for a single row only, in other words, for a single possible friend move,
    evaluate all possible enemy moves.

    Return a row of evaluation scores.
    """

    row = []
    for en_transition in en_transitions:
        curr_state = game_state.copy()
        curr_state.update(fr_transition, en_transition)
        score, _ = evaluate_state(curr_state)
        row.append(score)
    return row


def __row_scores_recursive(
    game_state: GameState, fr_transition, en_transitions, depth, cutoff_range, best_score):
    """
    Evaluate the scores for a single row only, in other words, for a single possible friend move,
    evaluate all possible enemy moves.

    Recursively call evaluation function if evaluated score is within range of the best score.

    Return a row of evaluation scores.
    """
    # TODO  It might be better to run this in two steps, calculate all scores for all rows first,
    #       then pick the top N scores and only do the recursive step on those

    row = []
    for en_transition in en_transitions:
        curr_state = game_state.copy()
        curr_state.update(fr_transition, en_transition)
        curr_score, _ = evaluate_state(curr_state)
        if curr_score > best_score:
            # Best score needs to be updated
            best_score = curr_score
        if curr_score > best_score - cutoff_range:
            # The best score is within range of the best score, make a new level and evaluate
            next_fr_transitions = curr_state.next_friend_transitions()
            _, curr_score = __evaluate_equilibrium_recursive(
                curr_state, next_fr_transitions, depth-1, cutoff_range)
        row.append(curr_score)
    return row, best_score


# def __evaluate_all_moves(game_state: GameState, fr_transitions, depth, cutoff_range):
#     weights = [1, 100, -10, -5, -1, -0.1]

#     # if depth == 0:
#     #     score, _ = evaluate_state(game_state)
#     #     return None, score

#     goal_score = goal_reward(game_state)
#     if goal_score is not None:
#         return None, goal_score * 200

#     en_transitions = game_state.next_enemy_transitions()

#     value_matrix = []

#     best_score = float("-inf")

#     for fr_transition in fr_transitions:
#         row = __row_scores(game_state, fr_transition, en_transitions)
#         # row = []
#         # for en_transition in en_transitions:
#         #     state_ij = game_state.copy()
#         #     state_ij.update(fr_transition, en_transition)
#         #     score_ij, _ = evaluate_state(state_ij, weights)
#         #     if score_ij > best_score:
#         #         best_score = score_ij
#         #     if score_ij > best_score - cutoff_range:
#         #         fr_transitions_ij = state_ij.next_friend_transitions()
#         #         _, score_ij = evaluate_all_moves(
#         #             state_ij, fr_transitions_ij, depth-1, cutoff_range)
#         #     row.append(score_ij)
#         value_matrix.append(row)

#     return solve_game(value_matrix)
