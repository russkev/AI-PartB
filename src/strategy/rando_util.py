"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""
from random import randrange, random
from state.game_state_fast import GameState


def random_move(state: GameState):
    """
    Randomly chooses a transitions from all possible
    """
    transitions = state.next_friend_transitions()
    return transitions[randrange(len(transitions))]

def biased_random_move(state: GameState, is_friend: bool):
    """
    Randomly chooses a transition but favours moving rather than throwing,
    if there are 3 current player tokens on the board, throwing is discouraged even more.
    """

    move_transitions = state.next_swing_slide_transitions(is_friend)
    throw_transitions = state.next_throw_transitions(is_friend)
    num_on_board = state.num_friends() if is_friend else state.num_enemies()

    if len(throw_transitions) == 0 and len(move_transitions) == 0:
        return None    
    if len(throw_transitions) == 0:
        return move_transitions[randrange(len(move_transitions))]
    elif len(move_transitions) == 0:
        return throw_transitions[randrange(len(throw_transitions))]
    else:
        throw_probability = 0.3 if num_on_board < 3 else 0.01
        if random() < throw_probability:
            return throw_transitions[randrange(len(throw_transitions))]
        else:
            return move_transitions[randrange(len(move_transitions))]
