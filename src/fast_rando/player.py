from random import randrange
from state.fast_game_state import GameState

class Player:

    def __init__(self, player):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "upper" (if the instance will
        play as Upper), or the string "lower" (if the instance will play
        as Lower).
        """
        self.game_state = GameState(upper=(player == "upper"), turn=0, friend_throws=0, enemy_throws=0, friends={}, enemies={})
    
    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        # # possible_moves = GameState.next_all_moves_for_side(
        # #     self.game_state.friends, self.game_state.friend_throws, self.game_state.is_upper
        # # )
        # possible_moves = self.game_state.next_friend_transitions()
        # # self.game_state.next_moves()
        # piece = possible_moves[randrange(len(possible_moves))]
        # return piece

        transitions = self.game_state.next_transitions_for_side(True)
        return transitions[randrange(len(transitions))]
        # return random_move(self.game_state)
    

    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        self.game_state.update(player_action, opponent_action)


