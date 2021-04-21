import math
import copy
import random
from state.game_state import GameState
from strategy.evaluation import eval_function

def minimax_paranoid_reduction(game_state):
    state_tree = build_state_tree(game_state)
    results = []
    for f_move in state_tree:
        results.append((f_move[0], min_layer(f_move, f_move[1])))

    return state_tree[max_layer(results)][0]

def min_layer(move, responses):
    scores = []
    for r in responses:
        scores.append(r[1])
    return min(scores)

def max_layer(moves):
    max_index = 0
    for i, m in enumerate(moves[1:]):
        if m[1] > moves[max_index][1]:
            max_index = i + 1
    return max_index

def build_state_tree(game_state: GameState):
    f_moves = game_state.next_friend_transitions()
    e_moves = game_state.next_enemy_transitions()
    minimax_tree = []

    random.shuffle(f_moves)
    random.shuffle(e_moves)

    for i, f_move in enumerate(f_moves):
        minimax_tree.append((f_moves[i], []))
        for j, e_move in enumerate(e_moves):
            # minimax_tree[i][1].append((e_moves[j],  eval_function(
            #     copy.deepcopy(game_state).simulate_moves(f_moves[i], e_moves[j]))))
            minimax_tree[i][1].append((e_moves[j],  eval_function(
                game_state.update(e_move, f_move))))

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
