import math
import copy
import random
from state.game_state import GameState
from strategy.evaluation import evaluate_state, evaluate_state_normalised
from strategy.ml_evaluation import evaluate
from heapq import heappush, heappop

def minimax_paranoid_reduction(game_state):
    state_tree = build_state_tree(game_state)
    results = []
    for f_move in state_tree:
        results.append((f_move[0], min_layer(f_move[1])))
    result = state_tree[max_layer(results)]
    return result[0]

def minimax_paranoid_reduction_2(game_state):
    # return build_state_tree_2(game_state)

    state_tree = build_state_tree_3(game_state)
    _, move, _ = state_tree[0]
    return move
    # state_tree = build_state_tree_2(game_state)
    # score, f_move = state_tree[0]
    # return f_move


    # results = []
    # for f_move in state_tree:
    #     score, move = heappop(f_move[1])
    #     heappush(results, (-score, f_move[0]))
    # negative_best_score, best_move,  = heappop(results)
    return best_move

def minimax_with_ml(game_state):
    state_tree = build_state_tree(game_state)
    results = []
    for f_move in state_tree:
        results.append((f_move[0], min_layer(f_move, f_move[1])))

    return state_tree[max_layer(results)][0]

def min_layer(responses):
    scores = []
    for response in responses:
        scores.append(response[1])
    return min(scores)

def max_layer(moves):
    max_index = 0
    for i, m in enumerate(moves[1:]):
        if m[1] > moves[max_index][1]:
            max_index = i + 1
    return max_index

def build_state_tree(game_state: GameState):
    f_moves = game_state.next_transitions_for_side(True)
    e_moves = game_state.next_transitions_for_side(False)
    minimax_tree = []

    random.shuffle(f_moves)
    random.shuffle(e_moves)

    for i, f_move in enumerate(f_moves):

        eval_offset = repeated_state_offset(game_state, f_move)

        minimax_tree.append((f_move, []))
        for _, e_move in enumerate(e_moves):
            game_state_ij = game_state.copy()
            game_state_ij.update(f_move, e_move)
            game_state_ij.branching = len(f_moves) * len(e_moves)
            eval_score = evaluate_state(game_state_ij)
            minimax_tree[i][1].append((e_move,  eval_score + eval_offset))

    return minimax_tree


def build_state_tree_2(game_state: GameState):
    f_moves = game_state.next_transitions_for_side(True)
    e_moves = game_state.next_transitions_for_side(False)
    # minimax_tree = []

    random.shuffle(f_moves)
    random.shuffle(e_moves)

    max_score = float("-inf")
    max_move = f_moves[0]
    for i, f_move in enumerate(f_moves):

        eval_offset = repeated_state_offset(game_state, f_move)

        # minimax_tree.append((f_move, []))
        row = []
        min_score = float("inf")
        for _, e_move in enumerate(e_moves):
            game_state_ij = game_state.copy()
            game_state_ij.update(f_move, e_move)
            game_state_ij.branching = len(f_moves) * len(e_moves)
            eval_score = evaluate_state(game_state_ij)
            if eval_score < min_score:
                min_score = eval_score
            # heappush(row, (eval_score + eval_offset, e_move))

        # score, _ = row[0]
        # heappush(minimax_tree, (-score, f_move))
        if min_score + eval_offset > max_score :
            max_score = min_score + eval_offset
            max_move = f_move

    return max_move

def build_state_tree_3(game_state: GameState):
    f_moves = game_state.next_transitions_for_side(True)
    e_moves = game_state.next_transitions_for_side(False)
    game_state.branching = len(f_moves) * len(e_moves)

    random.shuffle(f_moves)
    random.shuffle(e_moves)

    minimax_tree = []

    for f_move in f_moves:
        eval_offset = repeated_state_offset(game_state, f_move)
        min_row = []
        for e_move in e_moves:
            game_state_ij = game_state.copy()
            game_state_ij.update(f_move, e_move)
            eval_score = evaluate_state(game_state_ij)
            heappush(min_row, (eval_score, e_move))
        min_score, _ = min_row[0]
        heappush(minimax_tree, (-(min_score+eval_offset), f_move, min_row))
    
    return minimax_tree






def repeated_state_offset(game_state: GameState, fr_transition):
    temp_state = game_state.copy()
    temp_state.update(fr_transition)
    if temp_state.existing_moves.limit_is_close:
        return -400
    elif temp_state.existing_moves.limit_reached:
        return -2000
    else:
        return 0


def build_state_tree_with_ml(game_state: GameState):
    f_moves = game_state.next_transitions_for_side(True)
    e_moves = game_state.next_transitions_for_side(False)
    minimax_tree = []

    random.shuffle(f_moves)
    random.shuffle(e_moves)

    for i, f_move in enumerate(f_moves):
        minimax_tree.append((f_move, []))
        for j, e_move in enumerate(e_moves):
            game_state_ij = game_state.copy()
            game_state_ij.update(f_move, e_move)
            eval_score = evaluate(game_state_ij)
            minimax_tree[i][1].append((e_move,  eval_score))

    return minimax_tree







# def minimax(node, depth, maximising_player):
#     print(node)
#     if depth == 0:
#         return node
    
#     if maximising_player:
#         value = -math.inf
#         for move in node[1]:
#             value = max(value, minimax(move, depth - 1, False))
#         return value

#     else:
#         value = math.inf
#         for move in node[1]:
#             value = min(value, minimax(move, depth - 1, True))
#         return value

if __name__ == '__main__':
    # g = [
    #     ("move f 1", [("Move e 1", 2), ("Move e 2", 4)]),
    #     ("move f 2", [("Move e 1", -4), ("Move e 2", 6)]),
    #     ("move f 3", [("Move e 1", 3), ("Move e 2", 16)]),
    #     ("move f 4", [("Move e 1", 5), ("Move e 2", 5)]),
    #     ("move f 5", [("Move e 1", -10), ("Move e 2", -2)]),
    # ]
    
    turn = 0
    up_throws = 0
    low_throws = 0
    pos = {(0,0): [("up", "r"), ("up", "r")], (1,1): [("low", "r")], (1,2): [("low", "p")]}
    ups = {(0,0): [("up", "r"), ("up", "r")]}
    lows = {(1,1): [("low", "r")], (1,2): [("low", "p")]}

    game_state = (turn, up_throws, low_throws, pos, ups, lows)

    import time
    t_start = time.time()
    for i in range(10000):
        ups.copy()
    print('copy', time.time() - t_start)
    
    t_start = time.time()
    for i in range(10000):
        copy.deepcopy(ups)
    print('deep copy', time.time() - t_start)
