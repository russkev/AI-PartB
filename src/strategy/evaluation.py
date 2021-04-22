from state.game_state import GameState
from state.token import defeat_token
from state.location import distance
import numpy as np
from heapq import heappush, heappop


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
    num_kills = num_opponents_killed(game_state, is_friend=True)
    num_deaths = num_opponents_killed(game_state, is_friend=False)
    num_friend_useless, num_enemy_useless = num_useless(game_state)

    scores = [
        distance_to_killable_score(game_state),
        num_friend_useless,
        num_poss_move_killed,
        num_poss_throw_killed,
        num_kills,
        num_deaths,
    ]

    weights = [
        1,
        0,      #-7,
        0,      #-0.5,
        0,      #-0.2,
        100,    # 1
        -100    #-1
    ]

    final_scores = np.multiply(scores, weights)

    return np.dot(scores, weights), final_scores


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


def num_useless(game_state: GameState):
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

    friend_useless = max(f_rocks - e_scissors, 0) + max(f_papers - e_rocks, 0) + max(f_scissors - e_papers, 0)
    enemy_useless = e_rocks - f_scissors + e_papers - f_rocks + e_scissors - f_papers

    return friend_useless, enemy_useless


def distance_to_killable_score(game_state: GameState, is_friend):
    """
    Return a score based on the distance friend tokens are from the tokens they can kill.

    The score is higher when more pieces are closer to their targets

    Points are added for the closest killable token to an attacking friendly token only.
    If, for example a friendly rock is very near 2 enemy scissors and there is another friendly 
    rock that is much further away, a single score will be awarded for the closest pair only.
    """

    if is_friend:
        this_side_pieces = game_state.friends
        opponent_side_pieces = game_state.enemies
    else:
        this_side_pieces = game_state.enemies
        opponent_side_pieces = game_state.friends

    min_distances = []

    for (en_token, en_loc) in opponent_side_pieces:
        min_distance = 8
        min_fr_loc = None
        for (fr_token, fr_loc) in this_side_pieces:
            if en_token == defeat_token(fr_token):
                token_distance = distance(fr_loc, en_loc)
                if token_distance < min_distance:
                    min_distance = token_distance
                    min_fr_loc = fr_loc
        heappush(min_distances, (min_distance, min_fr_loc))

    used_fr_locs = set()
    return_distances = 0

    while len(min_distances) > 0:
        (min_distance, fr_loc) = heappop(min_distances)
        if fr_loc not in used_fr_locs:
            return_distances += 8 - min_distance
            used_fr_locs.add(fr_loc)

    return return_distances





    # for (f_token, f_loc) in game_state.friends:
    #     # min_distance = 8
    #     # min_loc = None
    #     for (e_token, e_loc) in game_state.enemies:
    #         if e_token == defeat_token(f_token):
    #             t_distance = distance(f_loc, e_loc)
    #             if t_distance < min_distance:
    #                 min_distance = t_distance
    #                 min_loc = e_loc
    #     if min_loc is not None:
    #         visited_enemies.add()
    #     friend_distance += 8 - min_distance
    #             # friend_defeat_found = True

    # # friend_distance = friend_distance if friend_defeat_found else 20
    # return friend_distance

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
