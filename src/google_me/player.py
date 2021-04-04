from strategy import random
from state.game_state import GameState

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
        if player == "lower":
            self.game_state.is_upper = False

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        # next_states = self.game_state.generate_random_move()
        
        # self.game_state.round += 1

        # return next_states[randrange(len(next_states))]


        return random.action(self.game_state)

        # return self.game_state.generate_random_move()
    
    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        self.game_state = self.game_state.update(opponent_action, player_action)
        # (_, t, p) = opponent_action
        # self.game_state.enemies.append((t, p))

