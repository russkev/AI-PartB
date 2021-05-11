"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""
from random import shuffle

from state.game_state import GameState


def book_first_four_moves(game_state: GameState):
    tokens = ['r','p','s']
    shuffle(tokens)
    if game_state.turn == 0:
        if game_state.is_upper:
            return ("THROW", tokens[0], (4,-2))
        else:
            return ("THROW", tokens[0], (-4,2))
    elif game_state.turn == 1:
        existing_tokens = [t[0] for t in game_state.friends.values()]
        available_tokens = [t for t in tokens if t not in existing_tokens]
        if game_state.is_upper:
            return ("THROW", available_tokens[0], (3, -2))
        else:
            return ("THROW", available_tokens[0], (-3, 1))
    elif game_state.turn == 2:
        existing_tokens = [t[0] for t in game_state.friends.values()]
        available_tokens = [t for t in tokens if t not in existing_tokens]
        if game_state.is_upper:
            return ("THROW", available_tokens[0], (3, -1))
        else:
            return ("THROW", available_tokens[0], (-3, 2))
    else:
        if game_state.is_upper:
            destinations = [(2,-2),(2,-1),(2,0)]
            shuffle(destinations)
            return ("SWING", (4,-2),destinations[0])
        else:
            destinations = [(-2, 0), (-2, 1), (-2, 2)]
            shuffle(destinations)
            return ("SWING", (-4, 2), destinations[0])

