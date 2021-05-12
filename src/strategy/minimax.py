import math
import copy
import random
from state.game_state import GameState
from strategy.evaluation import evaluate_state, 
from heapq import heappush

def minimax_paranoid_reduction(game_state):
    """
    Return the minimax result
    
    Use a slightly faster algorithm that doesn't build a complete state tree
    """
    return build_state_tree_fast(game_state)

def minimax_paranoid_reduction_tree(game_state):
    """
    Return the entire state tree
    """
    return build_state_tree(game_state)




def build_state_tree_fast(game_state: GameState):
    f_moves = game_state.next_transitions_for_side(True)
    e_moves = game_state.next_transitions_for_side(False)

    random.shuffle(f_moves)
    random.shuffle(e_moves)

    max_score = float("-inf")
    max_move = f_moves[0]
    for f_move in f_moves:

        eval_offset = repeated_state_offset(game_state, f_move)
        min_score = float("inf")

        for e_move in e_moves:
            game_state_ij = game_state.copy()
            game_state_ij.update(f_move, e_move)
            game_state_ij.branching = len(f_moves) * len(e_moves)
            eval_score = evaluate_state(game_state_ij)
            if eval_score < min_score:
                min_score = eval_score

        if min_score + eval_offset > max_score :
            max_score = min_score + eval_offset
            max_move = f_move

    return max_move

def build_state_tree(game_state: GameState):
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
