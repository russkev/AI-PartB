from enum import Enum
from copy import deepcopy
from itertools import product
from referee.game import Game
from state.board import Board
from state.location import distance
from state.token import defeat_token


class Phase(Enum):
    EARLY = 1
    MIDDLE = 2
    LATE = 3

class GameState:

    MAX_THROWS = 9
    MAX_TURNS = 360
    MAX_THROW_ENEMY_DISTANCE = 2

    slide_options = [(r, q) for r in [-1, 0, 1] for q in [-1, 0, 1] if (abs(r + q) < 2) and (r != 0 or q != 0)]
    board = Board(slide_options)

    def __init__(self, is_upper=True, turn=0, friend_throws=0, enemy_throws=0, friends=None, enemies=None):
        self.phase = Phase.EARLY
        self.is_upper = is_upper
        self.turn = turn
        self.friend_throws = friend_throws
        self.enemy_throws = enemy_throws
        self.friends = {} if friends is None else friends
        self.enemies = {} if enemies is None else enemies

    def update(self, friend_transition=None, enemy_transition=None):
        """ applies moves from both players to the game state, progressing the game one turn."""
        self.turn += 1
        self.__apply_move(friend_transition, True)
        self.__apply_move(enemy_transition, False)
        if friend_transition is not None:
            self.__battle(friend_transition[2])  # index 2 is the destination of the move
            if enemy_transition is not None and friend_transition[2] != enemy_transition[2]:
                self.__battle(enemy_transition[2]) # in the case friend and enemy don't move to the same location
        elif enemy_transition is not None:
            self.__battle(enemy_transition[2])

        if (self.friend_throws == 9) or (self.enemy_throws == 9):
            self.phase = Phase.LATE
        elif self.turn > 4:
            self.phase = Phase.MIDDLE

    def copy(self) -> "GameState":
        return GameState(self.is_upper, self.turn, self.friend_throws, self.enemy_throws, self.friends.copy(), self.enemies.copy())

    def num_friends(self):
        return self.num_in_play_for_side(is_friend=True)
    
    def num_enemies(self):
        return self.num_in_play_for_side(is_friend=False)

    def num_in_play_for_side(self, is_friend):
        reference = self.friends if is_friend else self.enemies
        count = 0
        for tokens in reference.values():
            count += len(tokens)
        return count

    def num_deaths(self):
        return self.friend_throws - self.num_friends()

    def num_kills(self):
        return self.enemy_throws - self.num_enemies()

    def next_transitions(self):
        """ all possible permutations of next moves from the game state."""
        return list(product(self.next_transitions_for_side(True), self.next_transitions_for_side(False)))

    def next_friend_transitions(self):
        return self.next_transitions_for_side(is_friend=True)
    
    def next_enemy_transitions(self):
        return self.next_transitions_for_side(is_friend=False)

    def next_transitions_for_side(self, is_friend):
        """ all possible moves for one side from the current game state."""
        return self.next_slide_transitions(is_friend) + self.next_swing_transitions(is_friend) + self.next_throw_transitions(is_friend)

    def next_swing_slide_transitions(self, is_friend):
        return self.next_slide_transitions(is_friend) + self.next_swing_transitions(is_friend)

    def __apply_move(self, move, is_friend):
        """ makes the move for a given side on the current game state."""
        if move is not None:
            reference = self.friends if is_friend else self.enemies
            if move[0] == 'THROW':
                # move indexes are move type, token, location
                reference[move[2]] = reference[move[2]] + [move[1]] if move[2] in reference else [move[1]]  # add token to location
                if is_friend: self.friend_throws += 1
                else: self.enemy_throws += 1

            else:
                # move indexes are move type, start location, end location
                # token = reference[move[1]].pop()  # get one of the tokens at the location
                token = GameState.__pop(reference, move[1])
                reference[move[2]] = reference[move[2]] + [token] if move[2] in reference else [token]  # add token to location
                

    def __pop(target_dict, location):
        """
        Pop an element from target_dict at location in a way that overwrites the original list
        (this prevents shallow copy issues)
        """
        removed = target_dict[location][0]
        target_dict[location] = target_dict[location][1:]
        if len(target_dict[location]) == 0:
            del target_dict[location]
        return removed


    def next_slide_transitions(self, is_friend):
        """ calculates all slide transitions for a side."""
        transitions = []
        reference = self.friends if is_friend else self.enemies
        for loc in reference.keys():
            for new_loc in GameState.board.get_slide_options(loc):
                transitions.append(("SLIDE", loc, new_loc))
        return transitions

    def next_swing_transitions(self, is_friend):
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

    def next_throw_transitions(self, is_friend):
        """ calculates all throw transitions for a side."""
        transitions = []
        throw_count = self.friend_throws if is_friend else self.enemy_throws
        if throw_count >= GameState.MAX_THROWS: return transitions # no more throw moves are allowed
        
        upper = (self.is_upper and is_friend) or ((not self.is_upper) and (not is_friend))
        for loc in GameState.board.get_throw_options(upper, throw_count):
            # any kind of pruning logic for throws should be imported and run here.
            for tok in ['r', 'p', 's']:
                transitions.append(('THROW', tok, loc))
        
        return self.__prune_throws(transitions, is_friend)

    def __prune_throws(self, throws, is_friend):
        """
        Prune throws that seem very unlikely to improve the score.

        If enemies aren't within 2 units of the farthest throw row, the throws only the throws up 
        to the farthest throw row are kept, otherwise only throws that throw a token to within 2
        units of a killable opponent are kept

        """
        # Enemies to the current throw side
        throw_enemies = self.enemies if is_friend else self.friends
        pruned_throws = []

        # Set num tokens used
        if is_friend:
            num_tokens_used = self.friend_throws
        else:
            num_tokens_used = self.enemy_throws

        # Append throws to the farthest row if there are no opponents in range
        if (is_friend and self.is_upper) or (not is_friend and not self.is_upper):
            # Upper player
            pruned_throws += GameState.__append_throws_distant(
                throws, num_tokens_used, throw_enemies, True)
        else:
            # Lower player
            pruned_throws += GameState.__append_throws_distant(
                throws, num_tokens_used, throw_enemies, False)

        # Append throws that are near opponents if opponent in range
        if len(pruned_throws) == 0:
            # There are opposing tokens less than MAX_DISTANCE away, add all throws within
            # MAX_THROW_ENEMY_DISTANCE of an enemy only.
            for throw in throws:
                (_, throw_token, throw_loc) = throw
                for enemy_loc, enemy_tokens, in throw_enemies.items():
                    if (distance(throw_loc, enemy_loc) <= GameState.MAX_THROW_ENEMY_DISTANCE
                            and enemy_tokens[0] == defeat_token(throw_token)):
                        pruned_throws.append(throw)
                        break

        return pruned_throws

    @staticmethod
    def __append_throws_distant(throws, num_tokens_used, throw_enemies, is_upper):
        """
        Return a list of throws to the farthest reachable row only if no enemy is closer, otherwise
        return an empty list.

        """

        farthest_r = GameState.farthest_r(num_tokens_used, is_upper)
        nearest_throw_enemy_r = -4 if is_upper else 4
        pruned_throws = []

        # Find row of nearest enemy row
        for (r, _) in throw_enemies.keys():
            if (is_upper and r > nearest_throw_enemy_r) or (not is_upper and r < nearest_throw_enemy_r):
                nearest_throw_enemy_r = r

        # Make pruned_throws all throws to the farthest reachable row only if the closest enemy
        # is further than that, otherwise make it empty
        if abs(farthest_r - nearest_throw_enemy_r) > GameState.MAX_THROW_ENEMY_DISTANCE:
            for throw in throws:
                (_, _, (throw_r, _)) = throw
                if throw_r == farthest_r:
                    pruned_throws.append(throw)
        return pruned_throws

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

    @staticmethod
    def farthest_r(num_tokens_used, is_upper):
        """
        Return the farthest reachable row for a side
        
        """
        if is_upper:
            return max(4 - num_tokens_used, -4)
        else:
            return min(-4 + num_tokens_used, 4)

    def moves_to_pieces(self, swing_slide_transitions, is_friend):
        """
        Convert a list of move commands of type: (type, from_loc, to_loc) to a list of pieces of
        type: (token, to_loc).
        """
        reference = self.friends if is_friend else self.enemies
        return_pieces = {}
        for (_, from_loc, to_loc) in swing_slide_transitions:
            for piece_loc, tokens in reference.items():
                if from_loc == piece_loc:
                    try:
                        return_pieces[to_loc].add(tokens[0])
                    except:
                        return_pieces[to_loc] = {tokens[0]}
                    # return_pieces.append((tokens[0], to_loc))
        return return_pieces


    def __str__(self):
        return f"Upper: {self.friends.__str__()}\nLower: {self.enemies.__str__()}\nTurn: {self.turn}\n"

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
