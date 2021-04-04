"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from itertools import product
import numpy as np
from random import randrange

from state.board import Board
from state.location import loc_add, loc_higher_than, loc_lower_than


class GameState:

    MAX_TOKENS = 9
    board = Board()

    slide_options = [(r, q) for r in [-1, 0, 1]
                     for q in [-1, 0, 1] if (abs(r + q) < 2) and (r != 0 or q != 0)]
    # the if condition (r != 0 or c != 0) means that the piece must move - this does not hold after part A.
    # abs condition makes sure the column and row change happens in the same turn, instead of sequentially.
    __slots__ = ("is_upper", "friends", "enemies", "costs", "hash",
                 "parent", "moves", "turn", "friend_waiting", "enemy_waiting")

    def __init__(self, *args, **kwargs):
        # self.board = Board()
        self.is_upper = True
        self.friends = []
        self.enemies = []
        self.hash = self.__hash__()
        self.costs = (0,0,0)
        self.parent = None
        self.moves = None
        self.turn = 0
        self.friend_waiting = 0
        self.enemy_waiting = 0
    
    def __copy_properties(self) -> "GameState":
        """
        Return a new `GameState` with all the relevant properties of the current
        state but with empty `friends` and `enemies` lists.
        """
        new_state = GameState()
        new_state.is_upper = self.is_upper
        new_state.turn = self.turn
        new_state.friend_waiting = self.friend_waiting
        return new_state

    def defeat_token(t):
        if t == 'r':
            return 's'
        elif t == 'p':
            return 'r'
        else:
            return 'p'

    def defeat_by_token(t):
        if t == 'r':
            return 'p'
        elif t == 'p':
            return 's'
        else:
            return 'r'
    
    def defeats(a_t, b_t):
        return (a_t == "r" and b_t == "s") \
            or (a_t == "p" and b_t == "r") \
            or (a_t == "s" and b_t == "p")


    def generate_random_move(self):
        """
        Pick a random move out of all the moves available to this `GameState`
        """

        # possible_moves = self.next_moves()
        possible_moves = GameState.next_moves_for_side(
            self.friends, 
            self.friend_waiting, 
            self.is_upper)
        self.next_moves()
        piece = possible_moves[randrange(len(possible_moves))]
        return piece

    def update(self, enemy_move, friend_move):
        """
        Main update function.

        Takes a friend and enemy move from player and returns a new GameState
        with updates applied, including any kills.
        """

        updated_pieces = self.__updated_pieces(friend_move, enemy_move)

        return self.__update_kills(updated_pieces)

        
    

    def __updated_pieces(self, friend_move, enemy_move):
        """
        Create a dict of type {location: [friendly piece], [enemy piece]} 
        with the state of all pieces after the move has been made but before 
        battles have been resolved.
        """
        updated_pieces = {}
        if friend_move[0] == "THROW":
            self.__update_throw(updated_pieces, friend_move, True)
        else:
            self.__update_swing_slide(updated_pieces, friend_move, True)
        if enemy_move[0] == "THROW":
            self.__update_throw(updated_pieces, enemy_move, False)
        else:
            self.__update_swing_slide(updated_pieces, enemy_move, False)
        return updated_pieces

    
    def __update_swing_slide(self, updated_pieces, new_move, is_friend):
        """
        Update `updated_pieces` with the the new pieces for a particular
        side after the swing or slide in `new_move` has been applied.

        `updated_pieces` should be a dict of the following type: 
            {location: [friendly piece], [enemy piece]}.

        `is_friend` flag used to determine if the friends or enemies are 
        updated.
        """
        found = False
        (_, prev_loc, new_loc) = new_move
        existing_pieces = self.friends if is_friend else self.enemies 
        for (t, existing_loc) in existing_pieces:
            if prev_loc == existing_loc and not found:
                found = True
                GameState.__append_loc_piece(
                    updated_pieces, (t, new_loc), is_friend)
            else:
                GameState.__append_loc_piece(
                    updated_pieces, (t, existing_loc), is_friend)

    
    def __update_throw(self, updated_pieces, new_move, is_friend):
        """
        Update `updated_pieces` with the the new pieces for a particular
        side after the throw in `new_move` has been applied.

        `updated_pieces` should be a dict of the following type: 
            {location: [friendly piece], [enemy piece]}.

        `is_friend` flag used to determine if the friends or enemies are 
        updated.
        """
        (_, new_t, new_loc) = new_move
        existing_pieces = self.friends if is_friend else self.enemies
        GameState.__append_loc_piece(updated_pieces, (new_t, new_loc), is_friend)
        for existing_piece in existing_pieces:
            GameState.__append_loc_piece(updated_pieces, existing_piece, is_friend)
        if is_friend:
            self.friend_waiting += 1

    
    def __append_loc_piece(board_pieces, piece, is_friend):
        """
        Add a new piece to piece's location in `board_pieces`.

        `board_pieces` should be a dict of the following type: 
            {location: [friendly piece], [enemy piece]}.

        `is_friend` flag used to determine if it is the friend or enemy piece
        list that needs to be updated.
        """
        (_, loc) = piece
        try:
            board_pieces[loc][int(not is_friend)].append(piece)
        except:
            board_pieces[loc] = ([],[])
            board_pieces[loc][int(not is_friend)].append(piece)


    def __update_kills(self, updated_pieces):
        """
        Run through all locations in `updated_pieces` and remove any pieces 
        that lose the battles at those locations.

        `board_pieces` should be a dict of the following type: 
            {location: [friendly piece], [enemy piece]}.

        Return a new GameState with all updated moves applied.
        """

        new_state = self.__copy_properties()
        new_state.turn += 1
        for loc, (fr_pieces, en_pieces) in updated_pieces.items():
            if len(fr_pieces) + len(en_pieces) > 1:

                condemmed = {GameState.defeat_token(
                    t) for (t, _) in fr_pieces + en_pieces}

                new_state.friends += [(t, loc)
                                for (t, _) in fr_pieces if t not in condemmed]
                new_state.enemies += [(t, loc)
                                for (t, _) in en_pieces if t not in condemmed]
            else:
                new_state.friends += fr_pieces
                new_state.enemies += en_pieces

        return new_state

    
    def next_moves(self):
        """
        Return all possible moves that can be reached from the current `GameState`

        """
        friend_moves = GameState.next_moves_for_side(
            self.friends, 
            self.friend_waiting, 
            self.is_upper)
        
        enemy_moves = GameState.next_moves_for_side(
            self.enemies,
            self.enemy_waiting,
            not self.is_upper
        )

        all_states = list(product(friend_moves, enemy_moves))

        return all_states
    
    def next_moves_for_side(pieces, num_tokens_waiting, is_upper):

        if num_tokens_waiting < GameState.MAX_TOKENS:
            moves = GameState.__throw_moves(num_tokens_waiting, is_upper)
        else:
            moves = []

        for (_, loc) in pieces:
            slide_moves = GameState.__slide_moves(loc)
            swing_moves = GameState.__swing_moves(pieces, loc, slide_moves)
            moves += slide_moves + swing_moves
        return moves


    # def set_state(self, pieces):
    #     """
    #     Sets the board state. Exclude blocks because they do not move and they are not needed after part A.
    #     """
    #     self.friends = pieces["upper"]
    #     self.enemies = pieces["lower"]

        # self.__piece_types()

    # def get_state(self):
    #     return {"upper": self.friends, "lower": self.enemies}

    def is_goal_state(self):
        """
        Return true if this GameState is in either a win or draw configuration
        """
        return len(self.enemies) == 0

    def __throw_moves(num_tokens_waiting, is_upper):
        """
        All possible throw moves

        Return tuple: ("THROW", token, location)
        """
        if is_upper:
            farthest_r = max(4 - num_tokens_waiting, -4)
            moves = [("THROW", t, loc) for t in ["r", "p", "s"] 
                        for loc in GameState.board.locations 
                        if not loc_lower_than(loc, farthest_r)]
        else:
            farthest_r = min(-4 + num_tokens_waiting, 4)
            moves = [("THROW", t, loc) for t in ["r", "p", "s"]
                        for loc in GameState.board.locations 
                        if not loc_higher_than(loc, farthest_r)]
        
        return moves


    def __slide_moves(loc):
        """
        All possible slide moves for a single location

        Return tuple: ("SLIDE", slide from location, slide to location)
        """

        result = []
        for d_loc in GameState.slide_options:
            if GameState.board.is_legal_location(loc_add(d_loc, loc)):
                result.append(("SLIDE", loc, loc_add(d_loc, loc)))
        

        return result

    def __swing_moves(pieces, curr_loc, slide_moves):
        """
        All possible swing moves for a single piece

        Return tuple: ("SWING", swing from location, swing to location)
        """
        result = []
        pivots = {loc for (_, _, loc) in slide_moves}


        for (_, other_loc) in pieces:
            # a piece will not swing around itself because its current 
            # location is not in the pivots (of the swing)
            if other_loc in pivots:
                for (_, _, swing_loc) in GameState.__slide_moves(other_loc):
                    if GameState.board.is_legal_location(swing_loc) \
                        and swing_loc not in pivots \
                        and swing_loc != curr_loc:
                        # swing destination must be a valid location, 
                        # somewhere that the piece could not slide to, 
                        # and cannot be the original piece location.
                        result.append(("SWING", curr_loc, swing_loc))

        return result

    # def __resolve_battles(self, moves):
    #     """
    #     go through all of the states and evaluate token battles to work out if piece are eliminating eachother.
    #     R beats S, S beats P, P beats R - regardless of which team the token is on.

    #     The first returned value is a list of lists. This is all of the possible positions for the upper pieces.
    #     The second returned value is the lower pieces. Only one instance of this is returned because there is not 
    #     movement in the lowers.
    #     """

    #     # TODO: improve the speed of this implementation
    #     # TODO: this code needs to be re-factored.
    #     states = [{}] * len(moves)

    #     # loop through all the states + the lower token pieces
    #     # find all the tokens that need to be eliminated from a given tile
    #     # loop through all of the states, deleting the tokens

    #     for i, move in enumerate(moves):
    #         occupied = self.__find_battles(move)

    #         # List of tuples of type: (r, q, tokens to eliminate)
    #         eliminations = {}
    #         for (r, q), (tokens, condemned) in occupied.items():
    #             elim = tokens.intersection(condemned)
    #             if len(elim) != 0:
    #                 eliminations[(r, q)] = elim

    #         if len(eliminations) == 0:
    #             states[i] = GameState(self.board, upper=list(move), lower=self.enemies)
    #         else:
    #             states[i] = GameState(self.board,
    #                 upper=GameState.__delete_pieces(move, eliminations),
    #                 lower=GameState.__delete_pieces(self.enemies, eliminations)
    #             )

    #     return states


    # def __find_battles(self, state):
    #     """
    #     Loops through all the tiles on the game board and determines which tokens need to be removed from each tile

    #     Return:
    #         Dictionary {Location (q, r) : (set of tokens at location, set of tokens to eliminate)}
    #     """

    #     # TODO: improve the speed of this implementation

    #     occupied = {} 

    #     # Loop though all friendly pieces in a provided potential board state
    #     # and mark their locations with their tokens
    #     for (t, r, q, *_) in state:
    #         if (r, q) not in occupied.keys():
    #             # occupiers and eliminate lists
    #             occupied[(r, q)] = (set(t), set())
    #         else:
    #             occupied[(r, q)][0].add(t)  # index 0 is the occupiers list

    #     # Loop though all enemy pieces and mark their locations with their tokens
    #     for (t, r, q) in self.enemies:  # need to add the lower
    #         if (r, q) not in occupied.keys():
    #             # occupiers and eliminate lists
    #             occupied[(r, q)] = (set(t), set())
    #         else:
    #             occupied[(r, q)][0].add(t)  # index 0 is the occupiers list


    #     for k in occupied.keys():
    #         # add the token type to eliminate to index 1
    #         for token in occupied[k][0]:
    #             if token == "s":
    #                 occupied[k][1].add("p")
    #             elif token == "p":
    #                 occupied[k][1].add("r")
    #             elif token == "r":
    #                 occupied[k][1].add("s")

    #     return occupied


    # @staticmethod
    # def __delete_pieces(state, eliminate):
    #     result = []
    #     for (t, r, q, *args) in state:  # token, location and move type
    #         eliminated = False
    #         try:
    #             eliminated = t in eliminate[(r, q)]
    #         except:
    #             pass
    #         if not eliminated:
    #             result.append((t, r, q, *args))
    #         # else:
    #         #     result.append((t + "x", r, q, *args))
    #     return result
    
    def __hash__(self):
        return (str(self.friends) + str(self.enemies)).__hash__()
    
    def __str__(self):
        return "Upper: {} | Costs: {}".format(self.friends.__str__(), self.costs)
        
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.hash == other.hash

    def __lt__(self, other):
        return self.costs[0] < other.costs[0]
