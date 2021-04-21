from state.game_state import GameState
from state.token import defeat_token
from state.location import distance
import numpy as np

def eval_function(game_state, enemy_transitions):
    """
    takes a game state and estimates the future utility.
    """
    # scores = [
    #     num_moves_difference(game_state),
    #     num_throws_difference(game_state),
    #     pieces_in_throw_range_difference(game_state),
    #     num_opponents_killed_difference(game_state),
    #     num_useless_difference(game_state),
    #     distance_to_killable(game_state),
    # ]

    # weights = [
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     -1,
    # ]

    num_poss_move_killed, num_poss_throw_killed = num_can_be_killed(game_state, enemy_transitions) 

    scores = [
        distance_to_killable(game_state),
        num_poss_move_killed,
        num_poss_throw_killed,
        num_opponents_killed(game_state, is_friend=True)
    ]

    weights = [
        -1,
        -0.5,
        -0.2,
        1
    ]

    final_scores = np.multiply(scores, weights)

    return np.dot(scores, weights)


def num_moves_difference(game_state: GameState):
    fr_moves = game_state.next_friend_moves()
    en_moves = game_state.next_enemy_moves()
    return len(fr_moves) - len(en_moves)


def num_throws_difference(game_state: GameState):
    return len(game_state.next_friend_throws()) - len(game_state.next_enemy_throws())


def pieces_in_throw_range_difference(game_state: GameState):

    return (pieces_in_throw_range(game_state, is_friend=True)
            - pieces_in_throw_range(game_state, is_friend=False))


def pieces_in_throw_range(game_state: GameState, is_friend):
    if is_friend:
        opponent_row = GameState.farthest_r(
            game_state.enemy_throws, not game_state.is_upper)
        pieces = game_state.friends
        is_upper = game_state.is_upper
    else:
        opponent_row = GameState.farthest_r(
            game_state.friend_throws, game_state.is_upper)
        pieces = game_state.enemies
        is_upper = not game_state.is_upper
    count = 0

    if is_upper:
        for (_, (r, _)) in pieces:
            if r <= opponent_row:
                count += 1

    else:
        for (_, (r, _)) in pieces:
            if r >= opponent_row:
                count += 1

    return count


def num_opponents_killed_difference(game_state: GameState):
    return game_state.num_kills() - game_state.num_deaths()
    # return (game_state.enemy_throws - len(game_state.enemies) - 
    #         (game_state.friend_throws - len(game_state.friends)))
    # # return (num_opponents_killed(game_state, is_friend=True)
    # #         - num_opponents_killed(game_state, is_friend=False))


def num_opponents_killed(game_state: GameState, is_friend):
    pieces = game_state.enemies if is_friend else game_state.friends
    throws = game_state.enemy_throws if is_friend else game_state.friend_throws
    return throws - len(pieces)


def num_useless_difference(game_state: GameState):
    f_rocks = f_papers = f_scissors = e_rocks = e_papers = e_scissors = 0

    for (token, _) in game_state.friends:
        if token == 'r':
            f_rocks += 1
        elif token == 'p':
            f_papers += 1
        else:
            f_scissors += 1
    for (token, _) in game_state.enemies:
        if token == 'r':
            e_rocks += 1
        elif token == 'p':
            e_papers += 1
        else:
            e_scissors += 1

    friend_useless = f_rocks - e_scissors + f_papers - e_rocks + f_scissors - e_papers
    enemy_useless = e_rocks - f_scissors + e_papers - f_rocks + e_scissors - f_papers

    return friend_useless - enemy_useless


def distance_to_killable(game_state: GameState):
    friend_distance = 0
    friend_defeat_found = enemy_defeat_found = False
    for (f_token, f_loc) in game_state.friends:
        for (e_token, e_loc) in game_state.enemies:
            if e_token == defeat_token(f_token):
                friend_distance += distance(f_loc, e_loc)
                friend_defeat_found = True

    friend_distance = friend_distance if friend_defeat_found else 20
    return friend_distance

def num_can_be_killed(game_state: GameState, enemy_transitions):
    move_killed_count = throw_killed_count = 0
    enemy_move_transitions = game_state.next_enemy_moves()
    for (_, fr_loc) in game_state.friends:
        for (transition_type, _, en_loc) in enemy_move_transitions:
            if fr_loc == en_loc:
                if transition_type == "THROW":
                    throw_killed_count += 1
                else:
                    move_killed_count += 1
    return move_killed_count, throw_killed_count