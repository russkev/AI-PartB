"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from random import randrange
from state.game_state import GameState
from strategy.rando_util import biased_random_move
from numpy.random import choice
from state.node_mcts_regret import Node

EXPLORATION_CONSTANT = 0.1

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

    if len(root.friend_transitions) == 0 and len(root.enemy_transitions) == 0:
        root.init_transitions()
    root.parent = None

    for _ in range(num_iterations):
        leaf = traverse(root)
        simulation_result = rollout(leaf)
        back_propagate(leaf, simulation_result)
    return select(root)


def traverse(node: Node):
    """
    Starting at the main root, traverse the tree. Use the UCT value to decide which child to visit
    in each step.

    Stop when a node is reached with children that have not been the root of a rollout.

    Pick a chiled from which to apply a rollout from the unvisited children.
    """

    while not node.has_unvisited_children():
        node = select(node)

    if node.goal_reward() is not None:
        return node

    return choose_unvisited(node)


def back_propagate(node: Node, reward):
    if node.parent is None:
        return
    node.num_visits += 1
    node.q_value += reward
    update_regret(node, reward)
    back_propagate(node.parent, reward)


####################################################################################################
#####-------------------------- END TRADITIONAL MCTS IMPLEMENTATION ---------------------------#####
####################################################################################################


def sm_mcts(root: Node, num_iterations) -> Node:
    """
    Entry point for Monte Carlo Tree Search algorithm as implemented in the following paper:
    https://papers.nips.cc/paper/2013/file/1579779b98ce9edb98dd85606f2c119d-Paper.pdf

    It functions the same way but is ordered a bit of a different way than the "traditional"
    approach
    """
    if len(root.friend_transitions) == 0 and len(root.enemy_transitions) == 0:
        root.init_transitions()

    for _ in range(num_iterations):
        iterate(root)

    child = select(root)
    return child


def iterate(root: Node):
    """
    Run a single iteration of the algorithm and update the root and any children of the root.
    """
    goal_reward = root.goal_reward()
    if goal_reward is not None:
        return goal_reward

    if root.has_unvisited_children():
        child = choose_unvisited(root)
        reward = rollout(child)
        child.q_value += reward
        child.num_visits += 1
        update_regret(root, reward)
        return return_value(reward, child.q_value, child.num_visits)

    else:
        child = select(root)
        reward = iterate(child)
        child.q_value += reward
        child.num_visits += 1
        update_regret(root, reward)
        return return_value(reward, child.q_value, child.num_visits)


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
            if goal_reward is not None:
                goal_reward = (goal_reward + 1) / 2
    return goal_reward


def rollout_policy(game_state: "Node") -> GameState:
    """
    Make random pair of moves (or use a very fast heuristic) and return the new state
    """
    friend_choice = biased_random_move(game_state, is_friend=True)
    enemy_choice = biased_random_move(game_state, is_friend=False)
    return game_state.update(enemy_choice, friend_choice)


def choose_unvisited(root: Node):
    available = [(i, j)
                 for i in range(len(root.friend_transitions))
                 for j in range(len(root.enemy_transitions))
                 if root.matrix[i][j] is None]

    (i, j) = available[randrange(len(available))]
    child = new_node(root, i, j)
    return child


def update_regret(root: Node, reward):
    """
    Update the regret array of the particular node, taking the new reward into account
    """
    for friend_index in range(len(root.friend_transitions)):
        root.regret[friend_index] += (regret_value(root, friend_index, reward) - reward)


def regret_value(node: Node, i, reward):
    """
    Return regret value based on friend index
    """
    if node.i == i:
        return reward
    else:
        return node.q_value


def select(root: Node):
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
        explore = EXPLORATION_CONSTANT * 1 / len(root.friend_transitions)
        exploit = (1-EXPLORATION_CONSTANT) * strategy(root, i)
        probabilities.append(explore + exploit)

    i = choice(range(len(root.friend_transitions)), p=probabilities)
    j = choice(range(len(root.enemy_transitions)))
    child = root.matrix[i][j]
    if child is None:
        child = new_node(root, i, j)
    return child


def strategy(root: Node, a):
    """
    Mixed strategy due to simultaneous play aspect

    Uses regret values to select probabilities for each move
    """
    regret_sum = 0
    for i in range(len(root.friend_transitions)):
        regret_sum += max(root.regret[i], 0)

    if regret_sum > 0:
        return max(root.regret[a], 0) / regret_sum
    else:
        return 1 / len(root.friend_transitions)


def return_value(reward, q_value, num_visits):
    """
    Return value.

    Not sure of the purpose of this since the number can be calculated at any point, 
    needs investigating
    """
    # TODO Investigate the purpose of this return value in the paper.
    return q_value / num_visits


def new_node(parent: Node, friend_index, enemy_index) -> Node:
    """
    Create a new node and initialize it.

    Connects parent and child, finds possible moves, etc.
    """
    friend_transition = parent.friend_transitions[friend_index]
    enemy_transition = parent.enemy_transitions[enemy_index]

    child = Node(parent.update(enemy_transition, friend_transition))
    child.parent = parent
    child.i = friend_index
    child.j = enemy_index
    parent.matrix[friend_index][enemy_index] = child

    child.init_transitions()
    child.action = (
        parent.friend_transitions[friend_index], parent.enemy_transitions[enemy_index])
    return child


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
    node.init_matrix()
    for _ in range(1000):
        iterate(node)

    ad = 98
