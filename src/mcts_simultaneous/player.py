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

class Player:

    def __init__(self, player):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "upper" (if the instance will
        play as Upper), or the string "lower" (if the instance will play
        as Lower).
        """
        game_state = GameState()
        self.root = Node(game_state)
        if player == "lower":
            self.root.is_upper = False

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """

        # test()
        random_turns = 20
        if self.root.turn < random_turns:
            return biased_random_move(self.root, is_friend=True)
        else:
            return monte_carlo_tree_search(self.root, 200).action[0]
            # return sm_mcts(self.root, 200).action[0]


    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        self.root = Node(self.root.update(opponent_action, player_action))


####################################################################################################
#####-------------------------------- NODE GAMESTATE WRAPPER ----------------------------------#####
####################################################################################################

class Node(GameState):
    MAX_DISTANCE = 2

    def __init__(self, other: GameState):
        """
        Initialise with a GameState.

        Set is_friend parameter to false since our starting position (root) will always be the case 
        where enemy has just moved and we're choosing where to move. This is making the (incorrect)
        assumption that moves are sequential instead of simultaneous.
        """
        super().__init__()
        self.is_upper = other.is_upper
        self.friends = other.friends
        self.enemies = other.enemies
        self.turn = other.turn
        self.friend_throws = other.friend_throws
        self.enemy_throws = other.enemy_throws

        self.action = None  # Action made to make this move
        self.parent = None
        # self.is_friend = False
        self.is_visited = False
        self.is_fully_expanded = False

        self.q_value = 0
        self.num_visits = 0

        self.friend_transitions = []
        self.enemy_transitions = []
        self.regret = []
        self.matrix = [[]]
        self.i = 0
        self.j = 0

    def init_transitions(self):
        self.friend_transitions = self.next_friend_moves()
        self.friend_transitions += self.__prune_throws(
            self.next_friend_throws(), True)

        self.enemy_transitions = self.next_enemy_moves()
        self.enemy_transitions += self.__prune_throws(
            self.next_enemy_throws(), False)

        self.init_matrix()

    def __prune_throws(self, throws, is_friend):
        throw_enemies = self.enemies if is_friend else self.friends
        pruned_throws = []

        if is_friend:
            tokens_remaining = self.friend_throws
        else:
            tokens_remaining = self.enemy_throws
        if (is_friend and self.is_upper) or (not is_friend and not self.is_upper):
            pruned_throws += Node.__append_throws_distant(
                throws, tokens_remaining, throw_enemies, True)
        else:
            pruned_throws += Node.__append_throws_distant(
                throws, tokens_remaining, throw_enemies, False)

        if len(pruned_throws) == 0:
            # There are opposing tokens less than MAX_DISTANCE away
            for throw in throws:
                (_, _, throw_loc) = throw
                for (_, enemy_loc) in throw_enemies:
                    if distance(throw_loc, enemy_loc) <= Node.MAX_DISTANCE:
                        pruned_throws.append(throw)
                        break

        return pruned_throws

    @staticmethod
    def __append_throws_distant(throws, tokens_remaining, throw_enemies, is_upper):
        # farthest_r = -4 if is_upper else 4
        farthest_r = GameState.farthest_r(tokens_remaining, is_upper)
        nearest_throw_enemy_r = -4 if is_upper else 4
        pruned_throws = []
        for (_, (r, _)) in throw_enemies:
            if r > nearest_throw_enemy_r:
                nearest_throw_enemy_r = r
        if abs(farthest_r - nearest_throw_enemy_r) > Node.MAX_DISTANCE:
            for throw in throws:
                (_, _, (throw_r, _)) = throw
                if throw_r == farthest_r:
                    pruned_throws.append(throw)
        return pruned_throws

    def init_matrix(self):
        self.matrix = [
            [None for _ in range(len(self.enemy_transitions))]
            for _ in range(len(self.friend_transitions))]

        for _ in range(len(self.friend_transitions)):
            self.regret.append(0)

    def has_unvisited_children(self):
        if self.is_fully_expanded:
            return False
        else:
            for friend_index in range(len(self.friend_transitions)):
                for enemy_index in range(len(self.enemy_transitions)):
                    if self.matrix[friend_index][enemy_index] is None:
                        return True
            self.is_fully_expanded = True
            return False

    def __repr__(self):
        return f"{self.q_value}/{self.num_visits}"


exploration_constant = 0.1

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
        explore = exploration_constant * 1 / len(root.friend_transitions)
        exploit = (1-exploration_constant) * strategy(root, i)
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
    child.action = (parent.friend_transitions[friend_index], parent.enemy_transitions[enemy_index])
    return child


def test():
    """
    Test function to investigate the MCTS code with a minimal possible actions.
    """
    state = GameState()
    node = Node(state)
    node.friend_throws = 9
    node.enemy_throws = 9
    node.friends = [('r', (-3, 0)), ('s', (4,0))]
    node.enemies = [('s', (-4, 0)), ('p', (0,4))]
    node.friend_transitions = node.next_friend_transitions()
    node.enemy_transitions = node.next_enemy_transitions()
    node.init_matrix()
    for _ in range(1000):
        iterate(node)

    ad = 98
