from heapq import heappush, heappop
from random import randrange
from time import time
from random import shuffle
from state.game_state import GameState
from strategy.rando_util import biased_random_move
import strategy.evaluation as eval
import numpy as np
from state.node_mcts_duct import Node
from strategy.mcts_duct import simple_reduction, monte_carlo_tree_search

def test_1():
    """
    Test function to investigate the MCTS code with a minimal possible actions.
    """
    state = GameState()
    node = Node(state)
    node.friend_throws = 9
    node.enemy_throws = 9

    node.friends = {(-3, 0): ['r'], (3, 1): ['s']}
    node.enemies = {(-4, 0): ['s'], (-4, 4): ['p']}
    result = simple_reduction(node)
    print(result.action)


def test_2():
    """
    *   throws:' `-.      ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *       | upper |    |       |       |  (P)  |       |       |
    *       |   5   |    |  4,-4 |  4,-3 |  4,-2 |  4,-1 |  4, 0 |
    *    ,-' `-._,-'  ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *   | lower |    |       |       |       |       |  (S)  |       |
    *   |   4   |    |  3,-4 |  3,-3 |  3,-2 |  3,-1 |  3, 0 |  3, 1 |
    *    `-._,-'  ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *            |       |       |       |       |       |       |       |
    *            |  2,-4 |  2,-3 |  2,-2 |  2,-1 |  2, 0 |  2, 1 |  2, 2 |
    *         ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *        |       |       |       |  (R)  |       |  (P)  |       |       |
    *        |  1,-4 |  1,-3 |  1,-2 |  1,-1 |  1, 0 |  1, 1 |  1, 2 |  1, 3 |
    *     ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *    |       |       |       |       |       |       |       |  (R)  |       |
    *    |  0,-4 |  0,-3 |  0,-2 |  0,-1 |  0, 0 |  0, 1 |  0, 2 |  0, 3 |  0, 4 |
    *     `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
    *        |       |       |       |       |       |       |  >s<  |       |
    *        | -1,-3 | -1,-2 | -1,-1 | -1, 0 | -1, 1 | -1, 2 | -1, 3 | -1, 4 |
    *         `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
    *            |  >s<  |  >p<  |       |       |       |       |       |
    *            | -2,-2 | -2,-1 | -2, 0 | -2, 1 | -2, 2 | -2, 3 | -2, 4 |
    *             `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
    *                |       |       |       |  >r<  |       |       |
    *                | -3,-1 | -3, 0 | -3, 1 | -3, 2 | -3, 3 | -3, 4 |
    *                 `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'   key:' `-.
    *                    |       |       |       |       |       |       | (sym) |
    *                    | -4, 0 | -4, 1 | -4, 2 | -4, 3 | -4, 4 |       |  r, q |
    *                     `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'         `-._,-'
    """
    state = GameState()
    node = Node(state)
    node.friend_throws = 5
    node.enemy_throws = 4
    node.friends = {(0, 3): ['r'], (1, -1): ['r'], (1, 1): ['p'], (3, 0): ['s'], (4, -2): ['p']}
    node.enemies = {(-3, 2): ['r'], (-2, -2): ['s'],
                    (-2, -1): ['p'], (-1, 3): ['s']}
    # simple_reduction(node)
    monte_carlo_tree_search(node, num_iterations=10,
                            playout_amount=3, node_cutoff=10)


def test_3():
    """
*   throws:' `-.      ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
*       | upper |    |       |       |       |       |       |
*       |   2   |    |  4,-4 |  4,-3 |  4,-2 |  4,-1 |  4, 0 |
*    ,-' `-._,-'  ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
*   | lower |    |       |       |  (R)  |       |       |       |
*   |   2   |    |  3,-4 |  3,-3 |  3,-2 |  3,-1 |  3, 0 |  3, 1 |
*    `-._,-'  ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
*            |       |  (P)  |       |       |       |       |       |
*            |  2,-4 |  2,-3 |  2,-2 |  2,-1 |  2, 0 |  2, 1 |  2, 2 |
*         ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
*        |  >s<  |       |       |       |       |       |       |       |
*        |  1,-4 |  1,-3 |  1,-2 |  1,-1 |  1, 0 |  1, 1 |  1, 2 |  1, 3 |
*     ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
*    |       |       |       |       |       |       |       |       |       |
*    |  0,-4 |  0,-3 |  0,-2 |  0,-1 |  0, 0 |  0, 1 |  0, 2 |  0, 3 |  0, 4 |
*     `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
*        |       |       |       |       |       |       |       |       |
*        | -1,-3 | -1,-2 | -1,-1 | -1, 0 | -1, 1 | -1, 2 | -1, 3 | -1, 4 |
*         `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
*            |       |       |       |       |       |       |       |
*            | -2,-2 | -2,-1 | -2, 0 | -2, 1 | -2, 2 | -2, 3 | -2, 4 |
*             `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
*                |       |       |       |       |       |       |
*                | -3,-1 | -3, 0 | -3, 1 | -3, 2 | -3, 3 | -3, 4 |
*                 `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'   key:' `-.
*                    |  >r<  |       |       |       |       |       | (sym) |
*                    | -4, 0 | -4, 1 | -4, 2 | -4, 3 | -4, 4 |       |  r, q |
*                     `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'         `-._,-'
    """
    state = GameState()
    node = Node(state)
    node.friend_throws = 2
    node.enemy_throws = 2
    node.friends = {(2, -3): ['p'], (3, -2): ['r']}
    node.enemies = {(-4, 0): ['r'], (1, -4): ['s']}
    simple_reduction(node)
    monte_carlo_tree_search(node, num_iterations=300,
                            playout_amount=3, node_cutoff=5)


def test_4():
    """
    Problem: Even though scissors knows that the most likely move for the enemy is to 
    move to its hex (and would thus die if it stayed), it still moves to another hex
    *   throws:' `-.      ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *       | upper |    |       |       |  (P)  |       |       |
    *       |   3   |    |  4,-4 |  4,-3 |  4,-2 |  4,-1 |  4, 0 |
    *    ,-' `-._,-'  ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *   | lower |    |       |       |       |       |       |  (R)  |
    *   |   3   |    |  3,-4 |  3,-3 |  3,-2 |  3,-1 |  3, 0 |  3, 1 |
    *    `-._,-'  ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *            |       |       |       |       |       |       |       |
    *            |  2,-4 |  2,-3 |  2,-2 |  2,-1 |  2, 0 |  2, 1 |  2, 2 |
    *         ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *        |       |       |       |       |       |       |       |       |
    *        |  1,-4 |  1,-3 |  1,-2 |  1,-1 |  1, 0 |  1, 1 |  1, 2 |  1, 3 |
    *     ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *    |       |       |  >s<  |       |       |       |       |       |       |
    *    |  0,-4 |  0,-3 |  0,-2 |  0,-1 |  0, 0 |  0, 1 |  0, 2 |  0, 3 |  0, 4 |
    *     `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
    *        |       |       |       |       |       |       |       |       |
    *        | -1,-3 | -1,-2 | -1,-1 | -1, 0 | -1, 1 | -1, 2 | -1, 3 | -1, 4 |
    *         `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
    *            |       |       |       |  >r<  |       |       |       |
    *            | -2,-2 | -2,-1 | -2, 0 | -2, 1 | -2, 2 | -2, 3 | -2, 4 |
    *             `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
    *                |       |       |  (S)  |       |       |       |
    *                | -3,-1 | -3, 0 | -3, 1 | -3, 2 | -3, 3 | -3, 4 |
    *                 `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'   key:' `-.
    *                    |       |  >p<  |       |       |       |       | (sym) |
    *                    | -4, 0 | -4, 1 | -4, 2 | -4, 3 | -4, 4 |       |  r, q |
    *                     `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'         `-._,-'
    """
    state = GameState()
    node = Node(state)
    node.friend_throws = 3
    node.enemy_throws = 3
    node.friends = {(4,-2):['p'], (3,1):['r'],(-3,1):['s']}
    node.enemies = {(0,-2):['s'], (-4,1):['p'], (-2,1):['r']}
    monte_carlo_tree_search(
        node,
        playout_amount=3,
        node_cutoff=3,
        outer_cutoff=7,
        num_iterations=900,
        # turn_time=2,
        exploration_constant=0.8,
        use_slow_culling=False,
        verbosity=2,
    )


def test_5():
    """
    Problem: players stay as far away as possible from the tokens they are supposed to attack
    *
    *   throws:' `-.      ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *       | upper |    |       |       |       |  >r<  |       |
    *       |   9   |    |  4,-4 |  4,-3 |  4,-2 |  4,-1 |  4, 0 |
    *    ,-' `-._,-'  ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *   | lower |    |       |       |       |       |       |       |
    *   |   9   |    |  3,-4 |  3,-3 |  3,-2 |  3,-1 |  3, 0 |  3, 1 |
    *    `-._,-'  ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *            |       |  (R)  |       |       |       |       |       |
    *            |  2,-4 |  2,-3 |  2,-2 |  2,-1 |  2, 0 |  2, 1 |  2, 2 |
    *         ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *        |       |       |  (R)  |       |       |       |  (S)  |       |
    *        |  1,-4 |  1,-3 |  1,-2 |  1,-1 |  1, 0 |  1, 1 |  1, 2 |  1, 3 |
    *     ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
    *    |       |       |       |       |       |       |       |       |       |
    *    |  0,-4 |  0,-3 |  0,-2 |  0,-1 |  0, 0 |  0, 1 |  0, 2 |  0, 3 |  0, 4 |
    *     `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
    *        |       |       |       |       |       |       |       |       |
    *        | -1,-3 | -1,-2 | -1,-1 | -1, 0 | -1, 1 | -1, 2 | -1, 3 | -1, 4 |
    *         `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
    *            |  >p<  |       |       |       |       |       |       |
    *            | -2,-2 | -2,-1 | -2, 0 | -2, 1 | -2, 2 | -2, 3 | -2, 4 |
    *             `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
    *                |       |       |       |       |       |  >s<  |
    *                | -3,-1 | -3, 0 | -3, 1 | -3, 2 | -3, 3 | -3, 4 |
    *                 `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'   key:' `-.
    *                    |       |  (P)  |       |       |       |       | (sym) |
    *                    | -4, 0 | -4, 1 | -4, 2 | -4, 3 | -4, 4 |       |  r, q |
    *                     `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'         `-._,-'
    """
    state = GameState()
    node = Node(state)
    node.friend_throws = 9
    node.enemy_throws = 9
    node.friends = {(2, -3): ['r'], (1, -2): ['r'],(1, 2): ['s'], (-4, 1): ['p']}
    node.enemies = {(4,-1):['r'], (-2,-2):['p'], (-3,4):['s']}
    monte_carlo_tree_search(
        node,
        playout_amount=3,
        node_cutoff=3,
        num_iterations=900,
        turn_time=2,
        exploration_constant=0.8,
        use_slow_culling=False,
        verbosity=2,
    )
