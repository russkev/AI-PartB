"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""
from heapq import heappush
from state.game_state import GameState
from state.token import defeats
from state.location import distance

"""
For generic components of the greedy implementation
"""


def opponent_distance_scores(game_state: GameState):
    """
    Return a heap of type [(-distance_score, move)] such that popping from the heap will provide the
    state with the highest distance score.
    """
    queue = []
    for friend_move in game_state.next_friend_transitions():
        score = opponent_distance_score(game_state, friend_move, game_state.next_enemy_transitions())
        heappush(queue, (-score, friend_move))

    return queue

def opponent_distance_score(game_state: GameState, friend_move, next_enemy_moves):
    """
    Return the average score for a particular friend move and all possible enemy moves
    """
    avg_score = 0
    for enemy_move in next_enemy_moves:
        new_state = game_state.update(enemy_move, friend_move)
        avg_score += individual_distance_score(new_state)
    avg_score /= len(next_enemy_moves)
    return avg_score


def individual_distance_score(new_state: GameState):
    """
    Return the distance score for a particular game state.
    For every enemy, the nearest friend that can defeat it is found (if it exists).
    The distance from the two pieces is calculated and a score is derived.
    
    Scores for individual pairs range from 0 to 8.
    
    Higher scores mean closer together pieces.
    """
    max_dist = 8
    score = 0
    for (enemy_t, enemy_loc) in new_state.enemies:
        min_dist = max_dist
        for(friend_t, friend_loc) in new_state.friends:
            if defeats(friend_t, enemy_t):
                curr_dist = distance(friend_loc, enemy_loc)
                if curr_dist < min_dist:
                    min_dist = curr_dist
        score += max_dist - min_dist
    return score
