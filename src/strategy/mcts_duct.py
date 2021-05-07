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

Also referenced these web pages:

https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/
https://github.com/AdamStelmaszczyk/gtsa/blob/master/cpp/gtsa.hpp



Benchmarks:
mcts_duct vs kev_greedy_5, 10,000 iterations per turn:
Total: 32 | Wins: 15 | Draws: 14 | Losses: 3 | Average time: 13.5 minutes (810 seconds)

Upper wins: 15/35, with ratio: 0.43
Lower wins: 3/35, with ratio: 0.09
Draws: 17/35, with ratio: 0.49

mcts_duct vs kev_greedy_5, 1,000 iterations per turn:
Upper wins: 5/35, with ratio: 0.14
Lower wins: 17/35, with ratio: 0.49
Draws: 13/35, with ratio: 0.37
"""

from random import randrange
from time import time
from random import shuffle
from state.game_state import GameState
from strategy.rando_util import biased_random_move
import strategy.evaluation as eval
import numpy as np
from state.node_mcts_duct import Node
EXPLORATION_CONSTANT = 0.4 # np.sqrt(2)
DEBUG_MODE = False
USE_PRIOR = False
NUM_PRIOR_VISITS = 4
USE_PRUNING = True
NUM_TO_KEEP = 5
rollout_count = 0


def monte_carlo_tree_search(root: Node, num_iterations=1000, playout_amount=6, node_cutoff=100) -> Node:
    """
    Entry point for the Monte Carlo Tree Search. This could run for ever so either a timer or
    a maximum number of iterations must be used to provide a cutoff.
    """
    global rollout_count

    root.parent = None
    while (root.num_visits < num_iterations):
        # A leaf node of the current frontier, does not include nodes visited in the rollout stage
        leaf = traverse(root, node_cutoff)
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
    return friend_winner

def simple_reduction(root: Node):
    add_children(root, 1000)
    scores = []
    for i, _ in enumerate(root.matrix):
        _, _, evaluation_score = sum_stats(root, i, is_row=True)
        # scores.append(evaluation_score / len(root.matrix[0]))
        scores.append(evaluation_score)

    best_score = float("-inf")
    best_index = 0
    for i, curr_score in enumerate(scores):
        if curr_score > best_score:
            best_score = curr_score
            best_index = i
    return root.friend_transitions[best_index]

def print_stats(root: Node, friend_winner, enemy_winner, winning_node: Node):
    """
    Print various stats useful for debugging MCTS
    """

    print(f"* GLOBAL STATS")
    print(f"* ratio:            {root.q_value / root.num_visits}")
    print(f"* simulations:      {root.num_visits}")
    print(f"* rollout states:   {rollout_count}")
    print(f"* children:         {len(root.matrix) * len(root.matrix[0])}")
    print(f"* Exploration constant: {EXPLORATION_CONSTANT}")
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


def traverse(node: Node, node_cutoff):
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
        add_children(node, node_cutoff)

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

    final_score, _ = eval.evaluate_state(game_state)
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
    shuffle(row_indices)
    shuffle(col_indices)

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
    
    winning_node = node.matrix[best_i][best_j]
    return node.friend_transitions[best_i], node.enemy_transitions[best_j], winning_node


def pick_unvisited_child(node: "Node") -> Node:
    """
    From all the children of node, randomly choose one that has not been visited and return it.
    Possible to use a heuristic here instead of randomness.
    """

    if USE_PRIOR:
        unvisited = node.unvisited_children(NUM_PRIOR_VISITS)
    else:
        unvisited = node.unvisited_children()
    shuffle(unvisited)
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

    Select best friend move and bests enemy move seperately.

    UCT uses the sum of all scores and visits for a particular row / column instead of just the one 
    for a particular move.

    When the best UCT score has been selected for both the friend and enemy, the corresponding
    child is selected from the matrix.

    Best UCT is the one with the highest value for the friend and lowest value for the enemy
    """

    # Best UCT for friend
    best_uct_friend = float("-inf")
    best_row_index = 0
    row_indices = list(range(len(node.matrix)))
    shuffle(row_indices)

    for i in row_indices:
        row_score_sum, row_visit_sum, _ = sum_stats(node, i, is_row=True)
        uct = get_uct(node.num_visits, row_visit_sum, row_score_sum, EXPLORATION_CONSTANT)
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
        uct = get_uct(node.num_visits, col_visit_sum, col_score_sum, EXPLORATION_CONSTANT)
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


def add_children(node: "Node", node_cutoff):
    """
    Calculate all friend and enemy moves that can be reached from the node state. 
    
    Add them to the node matrix as children.
    """
    # TODO for USE_PRIOR, figure out whether the parent needs to be updated with the new number
    # of visits and the updated q_score
    if len(node.friend_transitions) == 0 and len(node.enemy_transitions) == 0:
        node.friend_transitions = node.next_friend_transitions()
        node.enemy_transitions = node.next_enemy_transitions()
        if USE_PRIOR:
            node.matrix = [
                [
                    make_updated_node_with_prior(node, friend_transition, enemy_transition, NUM_PRIOR_VISITS)
                        for enemy_transition in node.enemy_transitions
                ]
                for friend_transition in node.friend_transitions
            ]
            # If node is root, add the appropriate number of prior visits
            
            node.num_visits += len(node.matrix) * len(node.matrix[0])
        else:
            node.matrix = [
                [
                    make_updated_node_with_eval_score(node, friend_transition, enemy_transition)
                        for enemy_transition in node.enemy_transitions
                ] 
                for friend_transition in node.friend_transitions
            ]
        if USE_PRUNING:
            prune_children(node, node_cutoff)

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


def make_updated_node_with_prior(node: Node, friend_transition, enemy_transition, num_prior_visits):
    """
    Make new node and give it existing visited and q_value counts based on the evaluation function
    return.
    """
    updated_node = node.copy_node_state()
    updated_node.update(friend_transition, enemy_transition)
    updated_node.parent = node
    updated_node.evaluation_score, _ = eval.evaluate_state(updated_node)
    # sigmoid_val = sigmoid(evaluation, 0.005)
    tanh_val = np.tanh(updated_node.evaluation_score*0.005)
    updated_node.num_visits = num_prior_visits
    updated_node.q_value = int(tanh_val * num_prior_visits)
    
    return updated_node

def make_updated_node_with_eval_score(node: Node, friend_transition, enemy_transition):
    """
    Make new node and give it the evaluation score
    """
    updated_node = node.copy_node_state()
    updated_node.update(friend_transition, enemy_transition)
    updated_node.parent = node
    updated_node.evaluation_score, _ = eval.evaluate_state(updated_node)
    return updated_node

def sigmoid(x, b):
    """
    standard sigmoid function
    """
    return 1 / (1 + np.exp(-b * x))

def test():
    """
    Test function to investigate the MCTS code with a minimal possible actions.
    """
    state = GameState()
    node = Node(state)
    node.friend_throws = 9
    node.enemy_throws = 9
    # node.is_friend=True

    node.enemies = {(-3, 0):['r'], (3,1):['s']}
    node.friends = {(-4, 0):['s'], (-4,4):['p']}
    result = simple_reduction(node)
    print(result.action)
    # print("Hello, World!")
