from random import randrange
from state.game_state import GameState
from strategy.evaluation_features import EvaluationFeatures

class Player:

    def __init__(self, player):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "upper" (if the instance will
        play as Upper), or the string "lower" (if the instance will play
        as Lower).
        """
        self.game_state = GameState(is_upper=(player == "upper"), turn=0, friend_throws=0, enemy_throws=0, friends={}, enemies={})
        self.evaluation_features = EvaluationFeatures()
    
    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """

        transitions = self.game_state.next_transitions_for_side(True)
        return transitions[randrange(len(transitions))]
    

    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        self.game_state.update(player_action, opponent_action)
        
        self.evaluation_features.calculate_features(self.game_state)

        self.evaluation_features.to_vector()




