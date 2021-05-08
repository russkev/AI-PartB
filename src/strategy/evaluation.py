from state.game_state import GameState
from state.token import defeat_token
from state.location import distance
import numpy as np
from heapq import heappush, heappop
from time import time
from state.token import defeat_by_token

def evaluate_state_normalised(game_state: GameState):
    final_score, scores = evaluate_state(game_state)
    final_score = np.tanh(final_score)
    return final_score, scores

def evaluate_state(game_state: GameState, weights=None):
    """
    takes a game state and estimates the future utility.
    """
    # TODO Maybe an evaluation for the likelihood of opponent being invincible

    # Distance to killable pieces score (fast)
    dist_to_killable_score_friend = distance_to_killable_score(game_state, is_friend=True)
    dist_to_killable_score_enemy = distance_to_killable_score(game_state, is_friend=False)
    dist_to_killable_score_diff = dist_to_killable_score_friend - dist_to_killable_score_enemy

    # Number opponents killed (fast)
    num_killed_diff = num_opponents_killed_difference(game_state)

    # Number of useless pieces (fast)
    num_friend_useless, num_enemy_useless = num_useless(game_state)
    num_useless_diff = num_friend_useless - num_enemy_useless

    # Throw range (medium)
    pieces_in_throw_range_diff = pieces_in_throw_range_difference(game_state)

    # Number of pieces that could be killed with a single move of the opponent (slowest)
    friend_move_to_pieces = game_state.moves_to_pieces(
        game_state.next_swing_slide_transitions(is_friend=True), is_friend=True)
    enemy_move_to_pieces = game_state.moves_to_pieces(
        game_state.next_swing_slide_transitions(is_friend=False), is_friend=False)
    pieces_in_move_range_diff = num_can_be_move_killed_difference(
        game_state, friend_move_to_pieces, enemy_move_to_pieces)

    # Total distance of pieces from the throw line (slow)
    distance_from_safeline_diff = distance_from_safeline_difference(game_state)

    # Invincible player
    friend_is_invincible = __player_is_invincible(game_state, is_friend=True)
    enemy_is_invincible = __player_is_invincible(game_state, is_friend=False)
    invincible_diff = friend_is_invincible - enemy_is_invincible



    scores = [
        dist_to_killable_score_diff,
        num_killed_diff,
        num_useless_diff,
        pieces_in_throw_range_diff,
        pieces_in_move_range_diff,
        distance_from_safeline_diff,
        invincible_diff
    ]

    if weights is None:
        weights = [
            10,
            200,
            -15,
            -5,
            -1,
            -1,
            500,
        ]

    final_scores = np.multiply(scores, weights)

    return np.dot(scores, weights), final_scores

def greedy_choose(game_state: GameState, weights=None):
    friend_transitions = game_state.next_friend_transitions()

    queue = []
    for friend_transition in friend_transitions:
        # New game state based on possible friend transition (enemy pieces stay the same)
        new_state = game_state.copy()
        new_state.update(friend_transition=friend_transition)
        # new_state = game_state.update(friend_transition=friend_transition)

        # Find the evaluation score
        eval_score, scores = evaluate_state(new_state, weights)

        # Add to queue. Use negative of score as first element of tuple since it is a min heap
        # A tuple of the individual scores are also included here for debugging purposes only
        heappush(queue, (-1 * eval_score, tuple(scores), friend_transition))

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
        opponent_row = GameState.farthest_r(game_state.enemy_throws, not game_state.is_upper)
        opponent_throws = game_state.enemy_throws
        pieces = game_state.friends
        is_upper = game_state.is_upper

    else:
        opponent_row = GameState.farthest_r(game_state.friend_throws, game_state.is_upper)
        opponent_throws = game_state.friend_throws
        pieces = game_state.enemies
        is_upper = not game_state.is_upper
    count = 0

    # if (is_friend and game_state.is_upper) or (not is_friend and not game_state.is_upper):
    #     return 

    if is_upper and (opponent_throws < game_state.MAX_THROWS):
        for (r, _) in pieces.keys():
            if r <= opponent_row:
                count += 1

    elif (not is_upper) and (opponent_throws < game_state.MAX_THROWS):
        for (r, _) in pieces.keys():
            if r >= opponent_row:
                count += 1

    return count


def distance_from_safeline_difference(game_state):
    return (distance_from_safeline(game_state, is_friend=True) -
            distance_from_safeline(game_state, is_friend=False))


def distance_from_safeline(game_state: GameState, is_friend):
    if is_friend:
        safe_row = GameState.farthest_r(
            game_state.friend_throws, game_state.is_upper)
        reference = game_state.friends
        is_upper = game_state.is_upper
    else:
        safe_row = GameState.farthest_r(
            game_state.enemy_throws, not game_state.is_upper)
        reference = game_state.enemies
        is_upper = not game_state.is_upper

    total_distance = 0

    if is_upper:
        for (r, _) in reference.keys():
            total_distance += max(0, safe_row - r)
    else:
        for (r, _) in reference.keys():
            total_distance += max(0, r - safe_row)
    
    return total_distance

def num_opponents_killed_difference(game_state: GameState):
    return game_state.num_kills() - game_state.num_deaths()


def num_useless(game_state: GameState):
    f_rocks = f_papers = f_scissors = e_rocks = e_papers = e_scissors = 0

    for f_tokens in game_state.friends.values():
        if f_tokens[0] == 'r':
            f_rocks += len(f_tokens)
        elif f_tokens[0] == 'p':
            f_papers += len(f_tokens)
        else:
            f_scissors += len(f_tokens)
    for e_tokens in game_state.enemies.values():
        if e_tokens[0] == 'r':
            e_rocks += len(e_tokens)
        elif e_tokens[0] == 'p':
            e_papers += len(e_tokens)
        else:
            e_scissors += len(e_tokens)

    friend_useless = max(f_rocks - e_scissors, 0) + max(f_papers - e_rocks, 0) + max(f_scissors - e_papers, 0)
    enemy_useless = max(e_rocks - f_scissors, 0) + max(e_papers - f_rocks, 0) + max(e_scissors - f_papers, 0)

    # Discount useless if all tokens is out and there are more friends than enemies
    if game_state.enemy_throws == game_state.MAX_THROWS:
        num_friends = game_state.num_friends()
        num_enemies = game_state.num_enemies()
        if num_friends > num_enemies:
            friend_useless -= (num_friends - num_enemies)

    # Discount useless if all tokens are out and there are more enemies than friends
    if game_state.friend_throws == game_state.MAX_THROWS:
        num_friends = game_state.num_friends()
        num_enemies = game_state.num_enemies()
        if num_enemies > num_friends:
            enemy_useless -= (num_enemies - num_friends)

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

    for en_loc, en_tokens in opponent_side_pieces.items():
        min_distance = 8
        min_fr_loc = None
        for fr_loc, fr_tokens in this_side_pieces.items():
            if en_tokens[0] == defeat_token(fr_tokens[0]):
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


def num_can_be_move_killed_difference(game_state, friend_move_to_pieces, enemy_move_to_pieces):
    return (num_can_be_move_killed(game_state, enemy_move_to_pieces, is_friend=True) 
            -num_can_be_move_killed(game_state, friend_move_to_pieces, is_friend=False)) 

def num_can_be_move_killed(game_state: GameState, move_to_pieces, is_friend):

    reference = game_state.friends if is_friend else game_state.enemies
    count = 0

    for curr_loc, curr_tokens in reference.items():
        if curr_loc in move_to_pieces:
            if defeat_by_token(curr_tokens[0]) in move_to_pieces[curr_loc]:
                count += len(curr_tokens)
    return count

    # for curr_loc, curr_tokens in reference.items():
    #     if curr_loc in move_to_pieces:
    #         opponent_token = 
    #         curr_tokens[0] == defeat_token()
    #     # for (opponent_token, opponent_loc) in move_to_pieces:
    #     #     if curr_loc == opponent_loc and curr_tokens[0] == defeat_token(opponent_token):
    #     #         count += len(curr_tokens)
    # return count

def goal_reward(game_state: GameState):
    """
    Return False if goal state has not been reached
    Return 1 if friend has won
    Return 0 if a draw has occurred
    Return -1 if the enemy has won
    """
    # 0.    If both players have throws available, goal state definitely has not been reached

    if game_state.friend_throws < game_state.MAX_THROWS and game_state.enemy_throws < game_state.MAX_THROWS:
        return None

    # 1.    One player has no remaining throws and all of their tokens have been defeated:
    #       If the other player still has tokens or throws, declare that player the winner.
    #       Otherwise, declare a draw.

    friend_moves_are_available = __moves_are_available(game_state, is_friend=True)
    enemy_moves_are_available = __moves_are_available(game_state, is_friend=False)

    if friend_moves_are_available and not enemy_moves_are_available:
        return 1
    elif not friend_moves_are_available and enemy_moves_are_available:
        return -1
    elif not friend_moves_are_available and not enemy_moves_are_available:
        return 0

    # 2.    A token is invincible if it cannot be defeated by the opponent’s remaining tokens,
    #       and the opponent has no remaining throws. Both players have an invincible token:
    #       Declare a draw

    friend_is_invincible = __player_is_invincible(game_state, is_friend=True)
    enemy_is_invincible = __player_is_invincible(game_state, is_friend=False)

    if friend_is_invincible and enemy_is_invincible:
        return 0

    # 3.    One player has an invincible token (see condition 2) and the other has only one
    #       remaining token (not invincible): Declare the player with the invincible token the
    #       winner

    elif friend_is_invincible and not enemy_is_invincible and game_state.num_enemies() == 1:
        return 1
    elif not friend_is_invincible and enemy_is_invincible and game_state.num_friends() == 1:
        return -1

    # 4.    One game configuration (with the same number of tokens with each symbol and
    #       controlling player occupying each hex, and the same number of throws remaining
    #       for each player), occurs for a third time since the start of the game
    #       (not necessarily in succession): Declare draw

    # NOT IMPLEMENTED

    # 5.    The players have had their 360th turn without a winner being declared:
    #       Declare a draw.

    if game_state.turn == GameState.MAX_TURNS:
        return 0

    return None

def __moves_are_available(game_state: GameState, is_friend):
    """
    Return true if there are any moves or throws available to the player
    """
    if is_friend:
        return not (game_state.friend_throws == game_state.MAX_THROWS and len(game_state.friends) == 0)
    else:
        return not (game_state.enemy_throws == game_state.MAX_THROWS and len(game_state.enemies) == 0)


def __player_is_invincible(game_state: GameState, is_friend):
    """
    Return true if player has at least one token that it is impossible for the other side
    to kill
    """
    if is_friend and game_state.enemy_throws < GameState.MAX_THROWS:
        return False
    elif not is_friend and game_state.friend_throws < GameState.MAX_THROWS:
        return False
    friend_tokens = __tokens_on_board(game_state, is_friend=True)
    enemy_tokens = __tokens_on_board(game_state, is_friend=False)
    if len(friend_tokens) == 3 and len(enemy_tokens) == 3:
        return False
    if is_friend:
        for friend_token in friend_tokens:
            if defeat_by_token(friend_token) not in enemy_tokens:
                return True

        return False
    else:
        for enemy_token in enemy_tokens:
            if defeat_by_token(enemy_token) not in friend_tokens:
                return True
        return False

def __tokens_on_board(game_state: GameState, is_friend) -> set:
    """
    Return a set of all tokens currently on the board for a player
    """
    reference = game_state.friends if is_friend else game_state.enemies
    tokens_types = set()
    for tokens in reference.values():
        for token in tokens:
            tokens_types.add(token)
    return tokens_types
