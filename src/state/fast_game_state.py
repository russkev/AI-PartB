from itertools import product
from state.board import Board


class GameState:

    MAX_THROWS = 9
    # MAX_TURNS = 360
    slide_options = [(r, q) for r in [-1, 0, 1] for q in [-1, 0, 1] if (abs(r + q) < 2) and (r != 0 or q != 0)]
    board = Board(slide_options)

    def __init__(self, upper=True, turn=0, friend_throws=0, enemy_throws=0, friends={}, enemies={}):
        self.upper = upper
        self.turn = turn
        self.friend_throws = friend_throws
        self.enemy_throws = enemy_throws
        self.friends = friends
        self.enemies = enemies

    def update(self, friend_move, enemy_move):
        """ applies moves from both players to the game state, progressing the game one turn."""
        self.turn += 1
        self.__apply_move(friend_move, True)
        self.__apply_move(enemy_move, False)
        self.__battle(friend_move[2])  # index 2 is the destination of the move
        if friend_move[2] != enemy_move[2]:
            self.__battle(enemy_move[2]) # in the case friend and enemy don't move to the same location

    def simulate_update(self, friend_move, enemy_move):
        """ creates a new gamestate to lookahead moves and states, does not modify actual game state."""
        new_state = self.__make_copy()
        new_state.update(friend_move, enemy_move)
        return new_state

    def next_transitions(self):
        """ all possible permutations of next moves from the game state."""
        return list(product(self.next_transitions_for_side(True), self.next_transitions_for_side(False)))

    def next_transitions_for_side(self, is_friend):
        """ all possible moves for one side from the current game state."""
        return self.__slide_transitions(is_friend) + self.__swing_transitions(is_friend) + self.__throw_transitions(is_friend)

    def __apply_move(self, move, is_friend):
        """ makes the move for a given side on the current game state."""
        reference = self.friends if is_friend else self.enemies
        if move[0] == 'THROW':
            # move indexes are move type, token, location
            reference[move[2]] = reference[move[2]] + [move[1]] if move[2] in reference else [move[1]]  # add token to location
            if is_friend: self.friend_throws += 1
            else: self.enemy_throws += 1

        else:
            # move indexes are move type, start location, end location
            token = reference[move[1]].pop()  # get one of the tokens at the location
            reference[move[2]] = reference[move[2]] + [token] if move[2] in reference else [token]  # add token to location
            
            if len(reference[move[1]]) == 0: del reference[move[1]]  # there are no more tokens at the location

    def __slide_transitions(self, is_friend):
        """ calculates all slide transitions for a side."""
        transitions = []
        reference = self.friends if is_friend else self.enemies
        for loc in reference.keys():
            for new_loc in GameState.board.get_slide_options(loc):
                transitions.append(("SLIDE", loc, new_loc))
        return transitions

    def __swing_transitions(self, is_friend):
        """ calculates all swing transitions for a side."""
        transitions = []
        reference = self.friends if is_friend else self.enemies
        for loc in reference.keys():
            possible_pivots = GameState.board.get_slide_options(loc)
            for pivot in possible_pivots:
                if pivot in reference:  # there exists an ally that can be used for a swing
                    for new_loc in GameState.board.get_swing_options(loc, pivot):
                        transitions.append(("SWING", loc, new_loc))    

        return transitions

    def __throw_transitions(self, is_friend):
        """ calculates all throw transitions for a side."""
        transitions = []
        throw_count = self.friend_throws if is_friend else self.enemy_throws
        if throw_count >= GameState.MAX_THROWS: return transitions # no move throw moves are allowed
        
        upper = (self.upper and is_friend) or ((not self.upper) and (not is_friend))
        for loc in GameState.board.get_throw_options(upper, throw_count):
            # any kind of pruning logic for throws should be imported and run here.
            for tok in ['r', 'p', 's']:
                transitions.append(('THROW', tok, loc))
        
        return transitions

    def __battle(self, location):
        """ checks for balles in the locations that have changed from the prior game state."""
        tokens = set(self.friends.get(location, []) + self.enemies.get(location, []))
        if len(tokens) == 1:
            return # there is only 1 type of token at the location, so no battles
        if len(tokens) == 3:
            # there can only be 3 types of tokens if both sides move tokens to the location
            del self.friends[location] # if every type of token exists, they all are defeated
            del self.enemies[location]
            return
        
        if tokens == {'r', 's'}: defeated = 's'  # the scissors is defeated by the rock
        elif tokens == {'s', 'p'}: defeated = 'p'
        elif tokens == {'p', 'r'}: defeated = 'r'

        if location in self.friends: self.__clear_defeated(location, defeated, True)
        if location in self.enemies: self.__clear_defeated(location, defeated, False)
        return

    def __clear_defeated(self, location, defeated, is_friend):
        """ removes specific tokens from a given location and for a given side."""
        reference = self.friends if is_friend else self.enemies
        reference[location] = [tok for tok in reference[location] if tok != defeated]  # remove defeated
        if len(reference[location]) == 0:  # remove location reference if no more tokens are on location
            del reference[location]

    def __make_copy(self):
        return GameState(self.upper, self.turn, self.friend_throws, self.enemy_throws, self.friends.copy(), self.enemies.copy())

    def __str__(self):
        return "Turn: {}\nUpper: {}\nLower: {}".format(self.turn, self.friends.__str__(), self.enemies.__str__())

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    g = GameState()
    # upper_g = GameState(True)
    # lower_g = GameState(False)
    # g.friends[(4,0)] = ["r"]
    # g.next_transitions_for_side(True)
    # g.friends[(3,1)] = ["r"]

    # upper_g.next_transitions_for_side(True)

    # lower_g.next_transitions_for_side(True)

    # g.friends[(2,0)] = ['r', 'p']
    # g.enemies[(2,1)] = ['s']

    move_1 = ('THROW', 's', (4,0))
    move_2 = ('THROW', 'r', (4,0))
    move_3 = ('THROW', 'r', (4,0))
    move_4 = ('THROW', 'p', (-4,1))
    move_5 = ('SLIDE', (-4,1), (-3,1))
    move_6 = ('SWING', (-3,1), (-1,1))

    g.update(move_1, move_4)
    g.update(move_2, move_5)

    g.next_transitions_for_side(True)
    # g.friend_throws = 9
    # g.next_transitions_for_side(True)
