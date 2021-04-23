"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from random import randrange
from time import time
from state.location import distance
from state.game_state import GameState
from strategy.rando_util import biased_random_move
import numpy as np
from numpy.random import choice
from state.node_mcts import Node


# Algorithm from:
# https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/


EXPLORATION_CONSTANT = 0.1


def monte_carlo_tree_search(root: Node) -> Node:
    """
    Entry point for the Monte Carlo Tree Search. This could run for ever so either a timer or
    a maximum number of iterations must be used to provide a cutoff.
    """
    root.is_visited=True
    count = 0
    while (count < 1000):
        # A leaf node of the current frontier, does not include nodes visited in the rollout stage
        leaf = traverse(root)
        # Make random moves to terminal and record the win, lose or draw score
        simulation_result = rollout(leaf)
        # Update all nodes in the appropriate branch with the simulation result (again, branch does
        # not include any nodes visited in rollout stage)
        back_propagate(leaf, simulation_result)
        count += 1

    # Out of all the children of the root node choose the best one (i.e. the most visited one)
    return best_child(root)


def traverse(node: Node):
    """
    Starting at the main root, traverse the tree. Use the UCT value to decide which child to visit 
    in each step. 
    
    Stop when a node is reached with children that have not been the root of a rollout.

    Pick a chiled from which to apply a rollout from the unvisited children.
    """

    # Find a node that hasn't been fully expanded
    while node.is_fully_expanded:
        node = best_uct(node)

    if node.goal_reward() is not None:
        # Node is terminal
        return node
    else:
        add_children(node)

    return pick_unvisited_child(node)


def rollout(node: "Node"):
    """
    Recursively choose moves for both sides until a terminal state is reached.
    Once terminal state reached, return the score in relation to root.friend 
    (+1 for win, 0 for draw, -1 for lose) 
    """
    goal_reward = node.goal_reward()
    if goal_reward is not None:
        return goal_reward
    else:
        game_state = rollout_policy(node)
        while goal_reward is None:
            game_state = rollout_policy(game_state)
            goal_reward = game_state.goal_reward()
    return goal_reward


def rollout_policy(game_state: "Node") -> GameState:
    """
    Make random pair of moves (or use a very fast heuristic) and return the new state
    """
    friend_choice = biased_random_move(game_state, is_friend=True)
    enemy_choice = biased_random_move(game_state, is_friend=False)
    return game_state.update(enemy_choice, friend_choice)


def back_propagate(node: "Node", result):
    """
    For all nodes that formed part of the branch which just had a rollout, 
    update the statistics for those nodes
    """
    node.num_visits += 1
    node.q_value += result if node.is_friend else -result
    if node.parent is None:
        return
    back_propagate(node.parent, result)


def best_child(node: Node):
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
        choice.is_visited = True
        if num_unvisited == 1:
            node.is_fully_expanded = True
        return choice


def best_uct(node: "Node"):
    """
    Choose the node that gives the highest UCT (Upper Confidence Bound for Trees) value.

    The exploration value is larger for nodes that have not yet been visited very many times. It 
    can be adjusted with the hyper-peramater: exploration_constant
    """
    best_child = None
    best_score = 0
    for child in node.children:
        exploitation = child.q_value / child.num_visits
        exploration = np.sqrt(np.log(node.num_visits) / child.num_visits)
        uct = exploitation + EXPLORATION_CONSTANT * exploration
        if uct > best_score:
            best_child = child
            best_score = uct
    
    return best_child


def add_children(node: "Node"):
    """
    Calculate all moves that can be reached from the node state. Add them to the node as children.

    Moves are just calculated for one side, namely the opposite side to the side defined by the 
    node.is_friend parameter
    """
    if len(node.children) == 0:
        if node.is_friend:
            child_moves = node.next_enemy_transitions()
        else:
            child_moves = node.next_friend_transitions()

        for child_move in child_moves:
            if node.is_friend:
                child = Node(node.update(enemy_transition=child_move))
            else:
                child = Node(node.update(friend_transition=child_move))
            child.parent = node
            child.is_friend = not node.is_friend
            child.action = child_move
            node.children.add(child)


def test():
    """
    Test function to investigate the MCTS code with a minimal possible actions.
    """
    state = GameState()
    node = Node(state)
    node.friend_throws = 9
    node.enemy_throws = 9
    node.friends = [('r', (-3, 0))]
    node.enemies = [('s', (-4, 0))]
    result = monte_carlo_tree_search(node)
    asad = 45
    print(result.action)
    # print("Hello, World!")
