"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088

Regret algorithm as described in this paper:
https://papers.nips.cc/paper/2013/file/1579779b98ce9edb98dd85606f2c119d-Paper.pdf
and here:
http://mlanctot.info/files/papers/cig14-smmctsggp.pdf
"""

from random import randrange
from state.game_state_fast import GameState
from strategy.rando_util import biased_random_move
from numpy.random import choice
from state.node_mcts_regret import Node
import strategy.evaluation as eval
import numpy as np

EXPLORATION_RATIO = 0.1

####################################################################################################
#####---------------------------- TRADITIONAL MCTS IMPLEMENTATION -----------------------------#####
####################################################################################################


def monte_carlo_tree_search(root: Node, num_iterations) -> Node:
    """
    Entry point for traditional implementation of the Monte Carlo Tree Search as implemented. 
    
    'Traditional' here means it clearly defines the Selection, Expansion, Simulation and 
    Backpropagation steps.
    
    The following pseudo code was used, with modifications for simultaneous move play.
    https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/
    """
    root.parent = None

    while (root.num_visits < num_iterations):
        # A leaf node of the current frontier, does not include nodes visited in the rollout stage
        leaf = traverse(root)
        # Make random moves to terminal and record the win, lose or draw score
        simulation_result = rollout(leaf)
        # Update all nodes in the appropriate branch with the simulation result (again, branch does
        # not include any nodes visited in rollout stage)
        back_propagate(leaf, simulation_result)
    # return get_best_child(root)
    return choose_winner(root)


def traverse(node: Node):
    """
    Starting at the main root, traverse the tree. Use the UCT value to decide which child to visit
    in each step.

    Stop when a node is reached with children that have not been the root of a rollout.

    Pick a chiled from which to apply a rollout from the unvisited children.
    """
    while node.is_fully_expanded:
        node = get_best_child(node)

    if eval.goal_reward(node) is not None:
        return node
    else:
        add_children(node)

    return pick_unvisited_child(node)


def back_propagate(node: Node, reward):
    """
    For all nodes that formed part of the branch which just had a rollout, 
    update the statistics for those nodes
    """
    update_stats(node, reward)
    if node.parent is not None:
        update_regret(node, reward)
        back_propagate(node.parent, reward)


def update_stats(node: Node, result):
    """
    Update stats of the given node
    """
    node.num_visits += 1
    node.q_value += result


def rollout(node: "Node"):
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
            game_state = rollout_policy(game_state)
            goal_reward = eval.goal_reward(game_state)
            if goal_reward is not None:
                goal_reward = (goal_reward + 1) / 2
    return goal_reward


def rollout_policy(game_state: GameState) -> GameState:
    """
    Make random pair of moves (or use a very fast heuristic) and return the new state
    """
    friend_choice = biased_random_move(game_state, is_friend=True)
    enemy_choice = biased_random_move(game_state, is_friend=False)
    new_state = game_state.copy()
    new_state.update(friend_choice, enemy_choice)
    return new_state


def pick_unvisited_child(node: Node) -> Node:
    """
    From all the children of node, randomly choose one that has not been visited and return it.
    Possible to use a heuristic here instead of randomness.
    """
    unvisited = node.unvisited_children()
    return unvisited[randrange(len(unvisited))]


def update_regret(node: Node, reward):
    """
    Update the regret array of the particular node, taking the new reward into account
    """
    for i in range(len(node.parent.matrix)):
        if i != node.i:
            node.parent.regret[i] += node.parent.matrix[i][node.j].q_value - reward



def get_best_child(root: Node):
    """
    Select an action using the following method:

    - Make a list of probabilities with each probability corresponding to a different action
    - Choose one of the actions based on those probabilities
    - Randomly choose a corresponding enemy action. (Note, this may not be the best idea)
    - If the chosen node is new, initialize it
    - Return the new node
    """
    # TODO Investigate the best way to choose an appropriate j value.
    probabilities = []
    for i in range(len(root.friend_transitions)):
        explore = EXPLORATION_RATIO * 1 / len(root.friend_transitions)
        exploit = (1-EXPLORATION_RATIO) * strategy(root, i)
        probabilities.append(explore + exploit)

    i = choice(range(len(root.friend_transitions)), p=probabilities)
    j = choice(range(len(root.enemy_transitions)))

    return root.matrix[i][j]

def choose_winner(root: Node):
    """
    Choose overall winning move based on mixed strategy output
    """
    # TODO These probabilities probably need to be a part of node and updated iteratively somewhere
    # else in the code
    probabilities = []
    for i in range(len(root.friend_transitions)):
        probabilities.append(strategy(root, i))
    
    i = choice(range(len(root.friend_transitions)), p=probabilities)

    return root.friend_transitions[i]


def strategy(root: Node, regret_index):
    """
    Mixed strategy due to simultaneous play aspect

    Uses regret values to select probabilities for each move
    """
    regret_sum = 0
    for i in range(len(root.friend_transitions)):
        regret_sum += max(root.regret[i], 0)

    if regret_sum > 0:
        return max(root.regret[regret_index], 0) / regret_sum
    else:
        return 1 / len(root.friend_transitions)


def add_children(node: Node):
    """
    Calculate all friend and enemy moves that can be reached from the node state. 
    
    Add them to the node matrix as children.
    """
    if len(node.friend_transitions) == 0 and len(node.enemy_transitions) == 0:
        node.friend_transitions = node.next_friend_transitions()
        node.enemy_transitions = node.next_enemy_transitions()
        node.regret = [0 for _ in range(len(node.friend_transitions))]
        node.matrix = [
            [
                node.make_updated_node(friend_transition, enemy_transition, parent=node, i=i, j=j)
                    for j, enemy_transition in enumerate(node.enemy_transitions)
            ]
            for i, friend_transition in enumerate(node.friend_transitions)
        ]


def test():
    """
    Test function to investigate the MCTS code with a minimal possible actions.
    """
    state = GameState()
    node = Node(state)
    node.friend_throws = 9
    node.enemy_throws = 9
    node.friends = [('r', (-3, 0)), ('s', (4, 0))]
    node.enemies = [('s', (-4, 0)), ('p', (0, 4))]
    node.friend_transitions = node.next_friend_transitions()
    node.enemy_transitions = node.next_enemy_transitions()

    ad = 98
