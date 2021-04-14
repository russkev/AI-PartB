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

class Node(GameState):
    def __init__(self, other: GameState):
        super().__init__()
        self.is_upper = other.is_upper
        self.friends = other.friends
        self.enemies = other.enemies
        self.turn = other.turn
        self.friend_throws = other.friend_throws
        self.enemy_throws = other.enemy_throws
        
        self.action = None
        self.parent = None
        self.children = set()
        self.is_friend = False
        self.is_visited = False
        self.is_fully_expanded = False

        self.q_value = 0
        self.num_visits = 0

    def unvisited_children(self):
        unvisited = []
        for child in self.children:
            if not child.is_visited:
                unvisited.append(child)
        return unvisited

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
        random_turns = 0
        if self.root.turn < random_turns:
            return biased_random_move(self.root, is_friend=True)
        else:
            return monte_carlo_tree_search(self.root).action


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
# https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/
def monte_carlo_tree_search(root) -> Node:
    # start_time = time()
    count = 0
    while (count < 100):
        leaf = traverse(root)
        simulation_result = rollout(leaf)
        back_propagate(leaf, simulation_result)
        count +=  1

    return best_child(root)


def traverse(node: Node):
    # fully_expand_node(node)
    # Find a node that hasn't been fully expanded
    while node.is_fully_expanded:
        node = best_uct(node)

    if node.goal_reward() is not None:
        # Node is terminal
        return node
    else:
        update_with_children(node)

    
    return pick_unvisited_child(node)


def rollout(node: "Node"):
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
    friend_choice = biased_random_move(game_state, is_friend=True)
    enemy_choice = biased_random_move(game_state, is_friend=False)
    # new_node = Node(curr_node.update(enemy_choice, friend_choice))
    # new_node.parent = curr_node
    # curr_node.children.add(new_node)
    return game_state.update(enemy_choice, friend_choice)


def back_propagate(node: "Node", result):
    if node.parent is None:
        return
    node.num_visits += 1
    node.q_value += result if node.is_friend else -result
    back_propagate(node.parent, result)


def best_child(node: Node):
    winner = None
    max_visits = 0
    for child in node.children:
        if child.num_visits > max_visits:
            max_visits = child.num_visits
            winner = child
    return winner



def pick_unvisited_child(node: "Node"):
    unvisited = node.unvisited_children()
    num_unvisited = len(unvisited)
    if num_unvisited == 0:
        node.is_fully_expanded = True
        return None
    else:
        choice = unvisited[randrange(num_unvisited)]
        choice.visited = True
        if num_unvisited == 1:
            node.is_fully_expanded = True
        return choice


def fully_expand_node(node: "Node"):
    transitions = node.next_enemy_transitions() if node.is_friend else node.next_friend_transitions()
    for transition in transitions:
        if node.is_friend:
            node.update(transition, None)
        else:
            node.update(None, transition)


def best_uct(node: "Node"):
    best_child = None
    best_score = 0
    exploration_constant = 0.5
    for child in node.children:
        exploitation = child.q_value / child.num_visits
        exploration = np.sqrt(np.log(node.num_visits) / child.num_visits)
        uct = exploitation + exploration_constant * exploration
        if uct > best_score:
            best_child = child
            best_score = uct
    return best_child

def update_with_children(node: "Node"):
    if len(node.children) == 0:
        if node.is_friend:
            child_moves = node.next_enemy_transitions()
        else:
            child_moves = node.next_friend_transitions()

        for child_move in child_moves:
            if node.is_friend:
                child = Node(node.update(enemy_move=child_move))
            else:
                child = Node(node.update(friend_move=child_move))
            child.parent = node
            child.is_friend = not node.is_friend
            child.action = child_move
            node.children.add(child)
