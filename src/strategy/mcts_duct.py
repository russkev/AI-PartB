"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088


Monte Carlo Tree Search with Decoupled Upper Confidence Bounds

Algorithm described here:
https://dke.maastrichtuniversity.nl/m.winands/documents/sm-tron-bnaic2013.pdf
and here:
http://mlanctot.info/files/papers/cig14-smmctsggp.pdf
and here:
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.704.9457&rep=rep1&type=pdf

UCT with prior knowledge:
https://hal.inria.fr/inria-00164003/document

Using evaluation functions in Monte Carlo Tree Search:
https://www.sciencedirect.com/science/article/pii/S0304397516302717
and here:
https://link.springer.com/content/pdf/10.1007%2F978-3-319-14923-3_4.pdf

Also referenced these web pages:
https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/
https://github.com/AdamStelmaszczyk/gtsa/blob/master/cpp/gtsa.hpp

"""

from heapq import heappush, heappop
from random import randrange
from time import time
from random import shuffle
from state.game_state import GameState
from strategy.rando_util import biased_random_move
import strategy.evaluation as eval
from strategy.minimax import minimax_paranoid_reduction_tree
import numpy as np
from state.node_mcts_duct import Node
DEBUG_MODE = False
# NUM_PRIOR_VISITS = 4
USE_PRUNING = True
NUM_TO_KEEP = 5
rollout_count = 0

exp_constant = 0.8 # np.sqrt(2)
start_time = end_time = time_consumed = 0
is_using_prior = False
is_using_fast_rollout_eval=False
is_using_fast_prune_eval=False
num_prior_visits = 4

def start_timer():
    global start_time
    start_time = time()

def end_timer():
    global start_time, end_time, time_consumed
    end_time = time()
    time_consumed += end_time - start_time



def monte_carlo_tree_search(
        root: Node, 
        playout_amount=6, 
        node_cutoff=100, 
        outer_cutoff=5, 
        num_iterations=float("inf"), 
        turn_time=float("inf"), 
        exploration_constant=0,
        use_slow_culling = False,
        verbosity=0,
        use_prior=True,
        num_priors=4,
        use_fast_rollout_eval=False,
        use_fast_prune_eval=False
    ) -> Node:
    """
    Entry point for the Monte Carlo Tree Search. This could run for ever so either a timer or
    a maximum number of iterations must be used to provide a cutoff.
    """


    global rollout_count, start_time, exp_constant, is_using_prior, num_prior_visits, \
        is_using_fast_prune_eval, is_using_fast_rollout_eval
    if exploration_constant != 0:
        exp_constant=exploration_constant
    start_time = time()
    root.parent = None
    is_using_prior = use_prior
    num_prior_visits = num_priors
    is_using_fast_rollout_eval = use_fast_rollout_eval
    is_using_fast_prune_eval = use_fast_prune_eval
    while (time() < start_time + turn_time and root.num_visits < num_iterations):
        # A leaf node of the current frontier, does not include nodes visited in the rollout stage
        leaf = traverse(root, node_cutoff, outer_cutoff, verbosity, use_slow_culling)
        # Make random moves to terminal and record the win, lose or draw score
        simulation_result = rollout(leaf, playout_amount)
        # Update all nodes in the appropriate branch with the simulation result (again, branch does
        # not include any nodes visited in rollout stage)
        back_propagate(leaf, simulation_result)
        # count += 1

    
    # Out of all the children of the root node choose the best one (i.e. the most visited one)
    friend_winner, enemy_winner, winning_node = choose_winner(root)

    if DEBUG_MODE:
        print_stats(root, friend_winner, enemy_winner, winning_node)

    if (verbosity >= 1):
        print(f"ITERATIONS: {root.num_visits}")
        print(f"TIME: {time() - start_time}")
    
    if (verbosity >= 2):
        print_stats(root, friend_winner, enemy_winner, winning_node)

    return friend_winner

def simple_reduction(root: Node):
    add_children(root, 6)
    fr_scores = []
    for i, _ in enumerate(root.matrix):
        _, _, evaluation_score = sum_stats(root, i, is_row=True)
        # scores.append(evaluation_score / len(root.matrix[0]))
        fr_scores.append(evaluation_score)

    en_scores = []
    for i, _ in enumerate(root.matrix[0]):
        _, _, evaluation_score = sum_stats(root, i, is_row=False)
        en_scores.append(evaluation_score)

    return root.friend_transitions[0]

def print_stats(root: Node, friend_winner, enemy_winner, winning_node: Node):
    """
    Print various stats useful for debugging MCTS
    """

    print(f"* GLOBAL STATS")
    print(f"* ratio:            {root.q_value / root.num_visits}")
    print(f"* simulations:      {root.num_visits}")
    print(f"* rollout states:   {rollout_count}")
    print(f"* children:         {len(root.matrix) * len(root.matrix[0])}")
    print(f"* Exploration constant: {exp_constant}")
    print(f"* WINNER STATS")
    if winning_node.num_visits > 0:
        ratio = winning_node.q_value / winning_node.num_visits
    else:
        ratio = 0
    
    print(f"* {winning_node.q_value:4} / {winning_node.num_visits:4}  "
          + f"{ratio:+.3f}"
          + f"  friend: {friend_winner}  enemy: {enemy_winner}")

    print(f"* CHILD STATS")

    print(f"* Score  | Visits | Ratio  |          Move")
    print(f"* -------+--------+--------+-----------------------------------")
    for i in range(len(root.matrix)):
        row_score, row_visits, _ = sum_stats(root, i, is_row=True)
        if row_visits > 0:
            ratio = row_score / row_visits
        else:
            ratio = 0
        print(f"* {row_score:+6} | {row_visits:+6} | {ratio:+.3f} | move: {root.friend_transitions[i]}")


def traverse(node: Node, node_cutoff, outer_cutoff, verbosity, use_slow_culling):
    """
    Starting at the main root, traverse the tree. Use the UCT value to decide which child to visit 
    in each step. 
    
    Stop when a node is reached with children that have not been the root of a rollout.
    Pick a chiled from which to apply a rollout from the unvisited children.
    """

    # Find a node that hasn't been fully expanded
    while node.is_fully_expanded:
        node = get_best_child(node)

    if eval.goal_reward(node) is not None:
        # Node is terminal
        return node
    else:
        add_children(node, node_cutoff, outer_cutoff, verbosity, use_slow_culling)

    return pick_unvisited_child(node)


def rollout(node: Node, playout_amount):
    """
    Recursively choose moves for both sides until a terminal state is reached.
    Once terminal state reached, return the score in relation to root.friend 
    (+1 for win, 0 for draw, -1 for lose) 
    """
    goal_reward = eval.goal_reward(node)
    if goal_reward is not None:
        return goal_reward
    else:
        game_state = rollout_policy(node)
        while goal_reward is None:
            if playout_amount == 0:
                goal_reward = evaluate_state_ternary(game_state)
                break
            game_state = rollout_policy(game_state)
            goal_reward = eval.goal_reward(game_state)
            playout_amount -= 1
    return goal_reward

def evaluate_state_ternary(game_state: GameState):
    """
    Return 
        +1  if evaluation function thinks a win is likely,
         0  if a draw is likely
        -1  if a lose is likely
    """
    global is_using_fast_rollout_eval

    if is_using_fast_rollout_eval:
        final_score = eval.evaluate_state_fast(game_state)
    else:
        final_score = eval.evaluate_state(game_state)
    if final_score > 0:
        return 1
    elif final_score == 0:
        return 0
    else:
        return -1


def rollout_policy(game_state: "Node") -> GameState:
    """
    Make random pair of moves (or use a very fast heuristic) and return the new state
    """
    global rollout_count
    rollout_count += 1
    friend_choice = biased_random_move(game_state, is_friend=True)
    enemy_choice = biased_random_move(game_state, is_friend=False)
    new_state = game_state.copy()
    new_state.update(friend_choice, enemy_choice)
    return new_state


def back_propagate(node: "Node", result):
    """
    For all nodes that formed part of the branch which just had a rollout, 
    update the statistics for those nodes
    """
    update_stats(node, result)
    if node.parent is not None:
        back_propagate(node.parent, result)


def update_stats(node: Node, result):
    """
    Update stats of the given node
    """
    node.num_visits += 1
    node.q_value += result

def sum_stats(node: Node, index, is_row=True):
    """
    Sum all visit and q_value stats for a given column or row

    index is the column or row to use

    is_row specifies whether it is a row

    """
    score_sum = 0
    visit_sum = 0
    eval_sum = 0
    if is_row:
        for j in range(len(node.matrix[0])):
            curr_node: Node = node.matrix[index][j]
            score_sum += curr_node.q_value
            visit_sum += curr_node.num_visits
            eval_sum += curr_node.evaluation_score
    else:
        for i in range(len(node.matrix)):
            curr_node: Node = node.matrix[i][index]
            score_sum += curr_node.q_value
            visit_sum += curr_node.num_visits
            eval_sum += curr_node.evaluation_score
    return score_sum, visit_sum, eval_sum



def choose_winner(node: Node):
    """
    Of all the children of node, choose the one with the best score. 
    Traditionally this is the one with the most visits.
    """

    best_friend_visits = 0
    best_enemy_visits = 0
    best_i = 0
    best_j = 0
    row_indices = list(range(len(node.matrix)))
    col_indices = list(range(len(node.matrix[0])))
    # shuffle(row_indices)
    # shuffle(col_indices)

    for i in row_indices:
        _, row_num_visits, _ = sum_stats(node, i, is_row=True)
        if row_num_visits > best_friend_visits:
            best_friend_visits = row_num_visits
            best_i = i
    
    for j in col_indices:
        _, col_num_visits, _ = sum_stats(node, j, is_row=False)
        if row_num_visits > best_enemy_visits:
            best_enemy_visits = col_num_visits
            best_j = j
    try:
        winning_node = node.matrix[best_i][best_j]
        friend_transition = node.friend_transitions[best_i]
        enemy_transition = node.enemy_transitions[best_j]
    except:
        # Occasionally matrix is empty
        winning_node = node.copy_node_state()
        try: 
            friend_transition = node.friend_transitions[0]
            enemy_transition = node.enemy_transitions[0]
            winning_node.update(friend_transition, enemy_transition)
        except:
            friend_transition = node.next_friend_transitions()[0]
            enemy_transition = node.next_enemy_transitions()[0]
            winning_node.update(friend_transition, enemy_transition)

    return friend_transition, enemy_transition, winning_node


def pick_unvisited_child(node: "Node") -> Node:
    """
    From all the children of node, randomly choose one that has not been visited and return it.
    Possible to use a heuristic here instead of randomness.
    """

    if is_using_prior:
        unvisited = node.unvisited_children(num_prior_visits * 2)
    else:
        unvisited = node.unvisited_children()
    # shuffle(unvisited)
    num_unvisited = len(unvisited)
    if num_unvisited == 0:
        node.is_fully_expanded = True
        return None
    else:
        choice = unvisited[randrange(num_unvisited)]
        if num_unvisited == 1:
            node.is_fully_expanded = True
        return choice


def get_best_child(node: Node):
    """
    Choose best child.

    Select best friend move and best enemy move seperately.

    UCT uses the sum of all scores and visits for a particular row / column instead of just the one 
    for a particular move.

    When the best UCT score has been selected for both the friend and enemy, the corresponding
    child is selected from the matrix.

    Best UCT is the one with the highest value for the friend and lowest value for the enemy
    """
    global exp_constant

    # Best UCT for friend
    best_uct_friend = float("-inf")
    best_row_index = 0
    row_indices = list(range(len(node.matrix)))
    shuffle(row_indices)

    for i in row_indices:
        row_score_sum, row_visit_sum, _ = sum_stats(node, i, is_row=True)
        uct = get_uct(node.num_visits, row_visit_sum, row_score_sum, exp_constant)
        if uct > best_uct_friend:
            best_uct_friend = uct
            best_row_index = i
    
    # Best UCT for enemy
    best_uct_enemy = float("inf")
    best_col_index = 0
    col_indices = list(range(len(node.matrix[0])))
    shuffle(col_indices)

    for j in col_indices:
        col_score_sum, col_visit_sum, _ = sum_stats(node, j, is_row=False)
        uct = get_uct(node.num_visits, col_visit_sum, col_score_sum, -exp_constant)
        if uct < best_uct_enemy:
            best_uct_enemy = uct
            best_col_index = j

    # Final choice
    return node.matrix[best_row_index][best_col_index]

                
def get_uct(parent_visits, visits, score, c):
    """
    Get the UCT score
    """

    return score / visits + c * np.sqrt(np.log(parent_visits) / visits)


def add_children(node: "Node", node_cutoff, outer_cutoff, verbosity, use_slow_culling):
    """
    Calculate all friend and enemy moves that can be reached from the node state. 
    
    Add them to the node matrix as children.
    """
    # TODO for USE_PRIOR, figure out whether the parent needs to be updated with the new number
    # of visits and the updated q_score
    global is_using_prior
    if len(node.friend_transitions) == 0 and len(node.enemy_transitions) == 0:
        # update_with_minimax(node, outer_cutoff)

        node.friend_transitions = node.next_friend_transitions()
        node.enemy_transitions = node.next_enemy_transitions()
        node.branching = len(node.friend_transitions) * len(node.enemy_transitions)
        prune_transitions(node, outer_cutoff)
        update_with_matrix_and_priors(node)
        
        # if use_slow_culling:
        #     update_with_pruned_matrix(node, node_cutoff)
        # elif is_using_prior:
        #     prune_transitions(node, outer_cutoff)
        #     update_with_matrix_and_priors(node)
        # else:
        #     prune_transitions(node, outer_cutoff)
        #     update_with_matrix_and_priors(node)

def update_with_minimax(node: Node, outer_cutoff):
    minimax_tree = minimax_paranoid_reduction_tree(node)
    minimax_tree_len = len(minimax_tree)
    node.matrix = []
    for i in range(min(outer_cutoff, minimax_tree_len)):
        fr_score, fr_transition, min_row = heappop(minimax_tree)
        # node.friend_transitions.append(fr_transition)
        min_row_len = len(min_row)
        node.branching = minimax_tree_len * min_row_len
        matrix_row = []
        for j in range(min(outer_cutoff, min_row_len)):
            score_ij, en_transition, node_ij = heappop(min_row)
            node_ij.parent = node
            update_priors(node_ij, score_ij)
            matrix_row.append(node_ij)
        node.matrix.append(matrix_row)
            # if j == 0:
                # node.enemy_transitions.append(en_transition)

    # new_fr_transitions = 

def prune_transitions(node: Node, outer_cutoff):
    fr_greedy_transition = eval.greedy_choose(node, is_friend=True)
    en_greedy_transition = eval.greedy_choose(node, is_friend=False)

    global is_using_fast_prune_eval
    evaluate_state_function = eval.evaluate_state_fast if is_using_fast_prune_eval else eval.evaluate_state

    # Shuffle so that states with equal scores have equal chance of being picked
    shuffle(node.friend_transitions)
    shuffle(node.enemy_transitions)

    fr_scores = []
    en_scores = []

    for i, fr_transition in enumerate(node.friend_transitions):
        state = node.copy()
        state.update(fr_transition, en_greedy_transition)
        score = score_with_repeated_state_check(node, evaluate_state_function(state))
        heappush(fr_scores, (-1 * score, i))
    
    for j, en_transition in enumerate(node.enemy_transitions):
        state = node.copy()
        state.update(fr_greedy_transition, en_transition)
        score = evaluate_state_function(state)
        heappush(en_scores, (+1 * score, j))
    
    new_fr_transitions = []
    new_en_transitions = []
    for _i in range(min(outer_cutoff, len(fr_scores))):
        _, fr_scores_index = heappop(fr_scores)
        new_fr_transitions.append(node.friend_transitions[fr_scores_index])
    for _j in range(min(outer_cutoff, len(en_scores))):
        _, en_scores_index = heappop(en_scores)
        new_en_transitions.append(node.enemy_transitions[en_scores_index])

    node.friend_transitions = new_fr_transitions
    node.enemy_transitions = new_en_transitions
        

def score_with_repeated_state_check(node: Node, score):
    new_score = score
    if node.existing_moves.limit_is_close:
        new_score -= 500
    if node.existing_moves.limit_reached:
        if score > 0:
            score-= 10000
        else:
            score -= 500
    return score

def prune_children(node: Node, node_cutoff):
    """
    Use the evaluation function to prune the friend and enemy moves under consideration.
    """
    friend_scores = []
    for i in range(len(node.matrix)):
        _, _, row_score = sum_stats(node, i, is_row=True)
        friend_scores.append((row_score, i))

    enemy_scores = []
    for j in range(len(node.matrix[0])):
        _, _, col_score = sum_stats(node, j, is_row=False)
        enemy_scores.append((col_score, j))

    friend_scores.sort(reverse=True)
    enemy_scores.sort(reverse=False)
    friend_scores = friend_scores[:node_cutoff]
    enemy_scores = enemy_scores[:node_cutoff]

    new_fr_transitions = [node.friend_transitions[x] for _, x in friend_scores]
    new_en_transitions = [node.enemy_transitions[x] for _, x in enemy_scores]

    new_matrix = [
        [
            node.matrix[i][j] for _, j in enemy_scores
        ]
        for _, i in friend_scores
    ]
    
    node.friend_transitions = new_fr_transitions
    node.enemy_transitions = new_en_transitions
    node.matrix = new_matrix


def update_with_matrix(node: Node):
    node.matrix = []
    for i in range(len(node.friend_transitions)):
        row = []
        for j in range(len(node.enemy_transitions)):
            updated_node = node.copy_node_state()
            updated_node.update(node.friend_transitions[i], node.enemy_transitions[j])
            updated_node.parent = node
            row.append(updated_node)
        node.matrix.append(row)


def update_with_matrix_and_priors(node: Node):
    node.matrix = []
    global is_using_fast_prune_eval
    evaluate_state_function = eval.evaluate_state_fast if is_using_fast_prune_eval else eval.evaluate_state

    for i in range(len(node.friend_transitions)):
        row = []
        for j in range(len(node.enemy_transitions)):
            updated_node = node.copy_node_state()
            updated_node.update(
                node.friend_transitions[i], node.enemy_transitions[j])
            updated_node.parent = node
            updated_node_score = evaluate_state_function(updated_node)
            update_priors(updated_node, updated_node_score)
            row.append(updated_node)
        node.matrix.append(row)

def update_priors(node: Node, score):
    global num_prior_visits
    tanh_score = np.tanh(score*0.005)
    node.num_visits = num_prior_visits
    node.q_value = tanh_score * num_prior_visits
    return

def update_with_pruned_matrix(node: Node, node_cutoff):

    fr_scores, en_scores = __get_prune_scores_slow(node)
    # fr_scores, en_scores = __get_prune_scores_quick(node)


    node.matrix = []
    new_fr_transitions = []
    new_en_transitions = []
    for i in range(min(node_cutoff, len(node.friend_transitions))):
        _, fr_tr_index = heappop(fr_scores)
        new_fr_transitions.append(node.friend_transitions[fr_tr_index])

        row = []
        for j in range(min(node_cutoff, len(node.enemy_transitions))):
            if i == 0:
                _, en_tr_index = heappop(en_scores)
                new_en_transitions.append(node.enemy_transitions[en_tr_index])
            updated_node = node.copy_node_state()
            updated_node.update(new_fr_transitions[i], new_en_transitions[j])
            updated_node.parent = node
            row.append(updated_node)
        node.matrix.append(row)

    node.friend_transitions = new_fr_transitions
    node.enemy_transitions = new_en_transitions


def __get_prune_scores_slow(node: Node):
    return (
        __get_prune_scores_slow_for_side(node, is_friend=True),
        __get_prune_scores_slow_for_side(node, is_friend=False)
    )


def __get_prune_scores_slow_for_side(node: Node, is_friend):
    ref_transitions = node.friend_transitions if is_friend else node.enemy_transitions
    multiplier = -1 if is_friend else 1
    scores = []
    global is_using_fast_prune_eval
    evaluate_state_function = eval.evaluate_state_fast if is_using_fast_prune_eval else eval.evaluate_state

    for i, ref_transition in enumerate(ref_transitions):
        updated_node = node.copy_node_state()
        if is_friend:
            temp_node = node.copy_node_state()
            temp_node.update(friend_transition=ref_transition)
            opp_transition = eval.greedy_choose(node, is_friend=False)
            updated_node.update(ref_transition, opp_transition)
        else:
            temp_node = node.copy_node_state()
            temp_node.update(enemy_transition=ref_transition)
            opp_transition = eval.greedy_choose(node, is_friend=True)
            updated_node.update(opp_transition, ref_transition)
        score = evaluate_state_function(updated_node)
        if is_friend:
            if updated_node.existing_moves.limit_is_close:
                score -= 500
            if updated_node.existing_moves.limit_reached:
                if score > 0:
                    score-= 10000
                else:
                    score -= 500        
        heappush(scores, (multiplier * score, i))
    return scores


def sigmoid(x, b):
    """
    standard sigmoid function
    """
    return 1 / (1 + np.exp(-b * x))

