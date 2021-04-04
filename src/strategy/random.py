from random import randrange
from state.game_state import GameState


def action(game_state):
    """
    Pick a random move out of all the moves available to this `GameState`
    """

    # possible_moves = self.next_moves()
    possible_moves = GameState.next_moves_for_side(game_state.friends, game_state.friend_waiting, game_state.is_upper)
    game_state.next_moves()
    piece = possible_moves[randrange(len(possible_moves))]
    return piece