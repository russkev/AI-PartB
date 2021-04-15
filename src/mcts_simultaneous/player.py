"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from random import randrange
from time import time
from state.game_state import GameState
from strategy.rando_util import biased_random_move
import numpy as np
from numpy.random import choice

class Node(GameState):
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

        test()
        # random_turns = 20
        # if self.root.turn < random_turns:
        #     return biased_random_move(self.root, is_friend=True)
        # else:
        #     return monte_carlo_tree_search(self.root).action


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

def sm_mcts(root: Node):
    
    goal_reward = root.goal_reward()
    if goal_reward is not None:
        return goal_reward

    if root.has_unvisited_children():
        (friend_index, enemy_index) = choose_unvisited(root)
        child = new_node(root, friend_index, enemy_index)
        reward = rollout(child)
        child.q_value += reward
        child.num_visits += 1
        update(root, friend_index, enemy_index, reward)
        return return_value(reward, child.q_value, child.num_visits)

    else:
        (friend_index, enemy_index) = select(root)
        child = root.matrix[friend_index][enemy_index]
        reward = sm_mcts(child)
        root.q_value += reward
        root.num_visits += 1
        update(root, friend_index, enemy_index, reward)
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
    return available[randrange(len(available))]

def update(root: Node, i, j, u1):
    for friend_index in range(len(root.friend_transitions)):
        root.regret[friend_index] += (x(root, i, friend_index, j, u1) - u1)

def x(h: Node, root_i, i, j, u1):
    if root_i == i:
        return u1
    else:
        return h.q_value


def select(root: Node):

    probabilities = []
    for i in range(len(root.friend_transitions)):
        # explore = exploration_constant * 1 / (len(root.friend_transitions) * len(root.enemy_transitions))
        explore = exploration_constant * 1 / len(root.friend_transitions)
        exploit = (1-exploration_constant) * strategy(root, i)
        probabilities.append(explore + exploit)

    
    sm = sum(probabilities)
    selected_index = choice(range(len(root.friend_transitions)), p=probabilities)
    return (selected_index, 0)

def strategy(root: Node, a):
    return 1/len(root.friend_transitions)
    # regret_sum = 0
    # for i in range(len(root.friend_transitions)):
    #     regret_sum += max(root.regret[i], 0)
    
    # if regret_sum > 0:
    #     return root.regret[a] / regret_sum
    # else:
    #     return 1 / len(root.regret)


def return_value(reward, q_value, num_visits):
    return q_value / num_visits

def new_node(parent: Node, friend_index, enemy_index) -> Node:
    friend_transition = parent.friend_transitions[friend_index]
    enemy_transition = parent.enemy_transitions[enemy_index]

    child = Node(parent.update(enemy_transition, friend_transition))
    child.parent = parent
    parent.matrix[friend_index][enemy_index] = child

    child.friend_transitions = child.next_friend_transitions()
    child.enemy_transitions = child.next_enemy_transitions()
    child.init_matrix()
    child.action = (parent.friend_transitions[friend_index], parent.enemy_transitions[enemy_index])
    return child


def test():
    state = GameState()
    node = Node(state)
    node.friend_throws = 9
    node.enemy_throws = 9
    node.friends = [('r', (-3, 0)), ('r', (4,0))]
    node.enemies = [('s', (-4, 0)), ('p', (0,4))]
    node.friend_transitions = node.next_friend_transitions()
    node.enemy_transitions = node.next_enemy_transitions()
    node.init_matrix()
    for _ in range(1000):
        sm_mcts(node)

    ad = 98
