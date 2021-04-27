"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from random import randrange
from time import time
from state.game_state_fast import GameState
from strategy.rando_util import biased_random_move
import strategy.evaluation as eval
import numpy as np
from state.node_mcts import Node
from strategy.evaluation import goal_reward



# Algorithm from:
# https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/
#
# Also referenced this algorithm:
# https://github.com/AdamStelmaszczyk/gtsa/blob/master/cpp/gtsa.hpp


EXPLORATION_CONSTANT = 0.9 # np.sqrt(2)
DEBUG_MODE = True
rollout_count = 0


def monte_carlo_tree_search(root: Node, num_iterations=1000, playout_amount=6) -> Node:
    """
    Entry point for the Monte Carlo Tree Search. This could run for ever so either a timer or
    a maximum number of iterations must be used to provide a cutoff.
    """
    global rollout_count

    root.parent = None
    while (root.num_visits < num_iterations):
        # A leaf node of the current frontier, does not include nodes visited in the rollout stage
        leaf = traverse(root)
        # Make random moves to terminal and record the win, lose or draw score
        simulation_result = rollout(leaf, playout_amount)
        # Update all nodes in the appropriate branch with the simulation result (again, branch does
        # not include any nodes visited in rollout stage)
        back_propagate(leaf, simulation_result)
        # count += 1

    if DEBUG_MODE:
        print_stats(root)
    # Out of all the children of the root node choose the best one (i.e. the most visited one)
    return choose_winner(root)

def print_stats(root: Node):
    print(f"* GLOBAL STATS")
    print(f"* ratio:            {root.q_value / root.num_visits}")
    print(f"* simulations:      {root.num_visits}")
    print(f"* rollout states:   {rollout_count}")
    print(f"* children:         {len(root.children)}")
    print(f"* Exploration constant: {EXPLORATION_CONSTANT}")
    print(f"* WINNER STATS")
    winner = choose_winner(root)
    print(f"* {winner.q_value:4} / {winner.num_visits:4}  {(winner.q_value / winner.num_visits):+.3f}  move: {winner.action}")
    print(f"* CHILD STATS")

    print(f"* Score  | Visits | Ratio  |          Move")
    print(f"* -------+--------+--------+-----------------------------------")
    for child in root.children:
        print(f"* {child.q_value:+6} | {child.num_visits:+6} | {(child.q_value / child.num_visits):+.3f} | move: {child.action}")

        # print(
        #     f"* score: {child.q_value} "
        #     + f"visits: {child.num_visits} "
        #     + f"UCT: {get_uct(child, EXPLORATION_CONSTANT):.3f} "
        #     + f"move: {child.action}"
        # )



def traverse(node: Node):
    """
    Starting at the main root, traverse the tree. Use the UCT value to decide which child to visit 
    in each step. 
    
    Stop when a node is reached with children that have not been the root of a rollout.
    Pick a chiled from which to apply a rollout from the unvisited children.
    """

    # Find a node that hasn't been fully expanded
    while node.is_fully_expanded:
        node = get_best_child(node)

    if goal_reward(node) is not None:
        # Node is terminal
        return node
    else:
        add_children(node)

    return pick_unvisited_child(node)


def rollout(node: "Node", playout_amount):
    """
    Recursively choose moves for both sides until a terminal state is reached.
    Once terminal state reached, return the score in relation to root.friend 
    (+1 for win, 0 for draw, -1 for lose) 
    """
    if node.is_friend:
        playout_amount += 1
    terminal_score = goal_reward(node)
    if terminal_score is not None:
        return terminal_score
    else:
        game_state = rollout_policy(node)
        while terminal_score is None:
            if playout_amount == 0:
                terminal_score = evaluate_state(game_state)
                break
            game_state = rollout_policy(game_state)
            terminal_score = eval.goal_reward(game_state)
            playout_amount -= 1
    return terminal_score

def evaluate_state(game_state: GameState):
    """
    Evaluate state. Assigning 1 if friend appears to be in a winning state
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

def choose_winner(node: Node):
    """
    Of all the children of node, choose the one with the best score. 
    Traditionally this is the one with the most visits.
    """
    winner = None
    max_visits = 0
    for child in node.children:
        if child.num_visits > max_visits:
            max_visits = child.num_visits
            winner = child
    return winner


def pick_unvisited_child(node: "Node") -> Node:
    """
    From all the children of node, randomly choose one that has not been visited and return it.
    Possible to use a heuristic here instead of randomness.
    """
    unvisited = node.unvisited_children()
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
    Get best child based on whether node is friend or enemy.
    """
    best_child = None
    if node.is_friend:
        best_uct = float("-inf")
        for child in node.children:
            uct = get_uct(child, EXPLORATION_CONSTANT)
            if (uct > best_uct):
                best_uct = uct
                best_child = child
    else:
        best_uct = float("inf")
        for child in node.children:
            uct = get_uct(child, -EXPLORATION_CONSTANT)
            if (uct < best_uct):
                best_uct = uct
                best_child = child
    return best_child
                
def get_uct(node: Node, c):
    """
    Return the UCT score for the particular state. 
    
    c is the exploration constant to be used 
    """
    parent_visits = 0
    if node.parent is not None:
        parent_visits = node.parent.num_visits
    score = node.q_value
    visits = node.num_visits

    return score / visits + c * np.sqrt(np.log(parent_visits) / visits)


def add_children(node: "Node"):
    """
    Calculate all moves that can be reached from the node state. Add them to the node as children.
    Moves are just calculated for one side, namely the opposite side to the side defined by the 
    node.is_friend parameter
    """
    if len(node.children) == 0:
        if node.is_friend:
            child_moves = node.next_friend_transitions()
        else:
            child_moves = node.next_enemy_transitions()

        for child_move in child_moves:
            child = Node(node.copy(), parent=node, is_friend=node.is_friend, action=child_move)
            if node.is_friend:
                child.update(friend_transition=child_move)
            else:
                child.update(enemy_transition=child_move)
            node.children.append(child)


def test():
    """
    Test function to investigate the MCTS code with a minimal possible actions.
    """
    state = GameState()
    node = Node(state)
    node.friend_throws = 9
    node.enemy_throws = 9
    node.is_friend=True
    node.friends = [('r', (-3, 0)), ('s', (3,1))]
    node.enemies = [('s', (-4, 0)), ('p', (-4,4))]
    result = monte_carlo_tree_search(node, num_iterations=20)
    asad = 45
    print(result.action)
    # print("Hello, World!")
