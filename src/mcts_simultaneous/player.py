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
        

        self.action = None # Action made to make this move
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

    def init_transitions(self):
        self.friend_transitions = self.next_friend_moves()
        self.friend_transitions += self.__prune_throws(self.next_friend_throws(), True)

        self.enemy_transitions = self.next_enemy_moves()
        self.enemy_transitions += self.__prune_throws(self.next_enemy_throws(), False)

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
            [ None for _ in range(len(self.enemy_transitions))] 
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
            # return monte_carlo_tree_search(self.root).action
            return mcts(self.root).action[0]


    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        self.root = Node(self.root.update(opponent_action, player_action))


# Algorithm from:
# https://papers.nips.cc/paper/2013/file/1579779b98ce9edb98dd85606f2c119d-Paper.pdf

exploration_constant = 0.1


def mcts(root: Node) -> Node:

    if len(root.friend_transitions) == 0 and len(root.enemy_transitions) == 0:
        root.init_transitions()

    for _ in range(2000):
        iterate(root)
    
    _, _, child = select(root)
    return child


def iterate(root: Node):
    """
    Simultaneous Monte Carlo Tree Search

    Run a single iteration of the algorithm and update the root and any children of the root.
    """
    goal_reward = root.goal_reward()
    if goal_reward is not None:
        return goal_reward

    if root.has_unvisited_children():
        friend_index, enemy_index, child = choose_unvisited(root)
        reward = rollout(child)
        child.q_value += reward
        child.num_visits += 1
        rm_update(root, friend_index, enemy_index, reward)
        return return_value(reward, child.q_value, child.num_visits)

    else:
        friend_index, enemy_index, child = select(root)
        reward = iterate(child)
        # root.q_value += reward
        # root.num_visits += 1
        # rm_update(root, friend_index, enemy_index, reward)
        # return return_value(reward, root.q_value, root.num_visits)
        child.q_value += reward
        child.num_visits += 1
        rm_update(root, friend_index, enemy_index, reward)
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
    return i, j, child


def rm_update(root: Node, i, j, reward):
    for friend_index in range(len(root.friend_transitions)):
        root.regret[friend_index] += (rm_regret_value(root, i, friend_index, j, reward) - reward)

def rm_regret_value(h: Node, root_i, i, j, u1):
    if root_i == i:
        return u1
    else:
        return h.q_value


def select(root: Node):
    probabilities = []
    for i in range(len(root.friend_transitions)):
        # for j in range(len(root.enemy_transitions)):
        # explore = exploration_constant * 1 / (len(root.friend_transitions) * len(root.enemy_transitions))
        explore = exploration_constant * 1 / len(root.friend_transitions)
        exploit = (1-exploration_constant) * strategy(root, i)
        probabilities.append(explore + exploit)

    
    sm = sum(probabilities)
    # selected_index = choice(range(len(root.friend_transitions) * len(root.enemy_transitions)), p=probabilities)
    # i = selected_index // len(root.friend_transitions)
    # j = selected_index % len(root.enemy_transitions)
    i = choice(range(len(root.friend_transitions)), p=probabilities)
    j = choice(range(len(root.enemy_transitions)))
    child = root.matrix[i][j]
    if child is None:
        child = new_node(root, i, j)
    return i, j, child

# def exp3_select(root: Node):
#     probabilities = []

#     for i in range(root.friend_transitions):
#         learning_rate = exploration_constant / len(root.friend_transitions)
#         exp = np.exp()

def strategy(root: Node, a):
    # return 1 / (len(root.friend_transitions) * len(root.enemy_transitions))
    regret_sum = 0
    for i in range(len(root.friend_transitions)):
        regret_sum += max(root.regret[i], 0)
    
    if regret_sum > 0:
        return max(root.regret[a], 0) / regret_sum
    else:
        return 1 / len(root.friend_transitions)
        # return 1 / (len(root.friend_transitions) * len(root.enemy_transitions))


def return_value(reward, q_value, num_visits):
    return q_value / num_visits

def new_node(parent: Node, friend_index, enemy_index) -> Node:
    friend_transition = parent.friend_transitions[friend_index]
    enemy_transition = parent.enemy_transitions[enemy_index]

    child = Node(parent.update(enemy_transition, friend_transition))
    child.parent = parent
    parent.matrix[friend_index][enemy_index] = child

    child.init_transitions()
    child.action = (parent.friend_transitions[friend_index], parent.enemy_transitions[enemy_index])
    return child


def test():
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
