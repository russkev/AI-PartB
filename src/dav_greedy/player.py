"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from state.game_state import GameState
from strategy.minimax import minimax_paranoid_reduction

class Player:

    def __init__(self, player):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "upper" (if the instance will
        play as Upper), or the string "lower" (if the instance will play
        as Lower).
        """
        self.game_state = GameState()        
        self.game_state.is_upper = player == "upper"

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        print("a")

        print("a")

        a = minimax_paranoid_reduction(self.game_state)

        action_state = self.game_state.copy_state()


        # do i build the tree here then pass it to minimax?

        self.game_state.next_transitions()[0:2]  # list that is [friend, enemy]

        return ('THROW', 'r', (4, 0))

        # return GameState.next_moves_for_side(
        #     self.game_state.friends, self.game_state.friend_throws, self.game_state.is_upper
        # )[0]
        
        # queue = opponent_distance_scores(self.game_state)
        # (best_score, best_move) = heappp(queue)
        # return best_move
        
    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        self.game_state = self.game_state.update(opponent_action, player_action)

    




