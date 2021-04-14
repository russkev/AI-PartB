"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from itertools import product

from state.board import Board
from state.location import loc_add, loc_higher_than, loc_lower_than
from state.token import defeat_by_token, defeat_token

class GameState:

    MAX_TOKENS = 9
    MAX_TURNS = 360
    board = Board()

    slide_options = [(r, q) for r in [-1, 0, 1]
                     for q in [-1, 0, 1] if (abs(r + q) < 2) and (r != 0 or q != 0)]


    __slots__ = ("is_upper", "friends", "enemies", "hash", "turn", "friend_throws", "enemy_throws")


    def __init__(self):
        self.is_upper = True
        self.friends = []
        self.enemies = []
        self.hash = self.__hash__()
        self.turn = 0
        self.friend_throws = 0
        self.enemy_throws = 0
    

    def __copy_properties(self) -> "GameState":
        """
        Return a new `GameState` with all the relevant properties of the current state but with 
        empty `friends` and `enemies` lists.
        """
        new_state = GameState()
        new_state.is_upper = self.is_upper
        new_state.turn = self.turn
        new_state.friend_throws = self.friend_throws
        new_state.enemy_throws = self.enemy_throws
        return new_state

    def goal_reward(self):
        """
        Return False if goal state has not been reached
        Return 1 if friend has won
        Return 0 if a draw has occurred
        Return -1 if the enemy has won
        """
        # 0.    If both players have throws available, goal state definitely has not been reached

        if self.friend_throws < self.MAX_TOKENS and self.enemy_throws < self.MAX_TOKENS:
            return None

        

        # 1.    One player has no remaining throws and all of their tokens have been defeated: 
        #       If the other player still has tokens or throws, declare that player the winner. 
        #       Otherwise, declare a draw.

        friend_moves_are_available = self.__moves_are_available(is_friend=True)
        enemy_moves_are_available = self.__moves_are_available(is_friend=False)

        if friend_moves_are_available and not enemy_moves_are_available:
            return 1
        elif not friend_moves_are_available and enemy_moves_are_available:
            return -1
        elif not friend_moves_are_available and not enemy_moves_are_available:
            return 0

        # 2.    A token is invincible if it cannot be defeated by the opponent’s remaining tokens, 
        #       and the opponent has no remaining throws. Both players have an invincible token: 
        #       Declare a draw

        friend_is_invincible = self.__player_is_invincible(is_friend=True)
        enemy_is_invincible = self.__player_is_invincible(is_friend=False)

        if friend_is_invincible and enemy_is_invincible:
            return 0


        # 3.    One player has an invincible token (see condition 2) and the other has only one 
        #       remaining token (not invincible): Declare the player with the invincible token the 
        #       winner

        elif friend_is_invincible and not enemy_is_invincible and len(self.enemies) == 1:
            return 1
        elif not friend_is_invincible and enemy_is_invincible and len(self.friends) == 1:
            return -1

        # 4.    One game configuration (with the same number of tokens with each symbol and 
        #       controlling player occupying each hex, and the same number of throws remaining 
        #       for each player), occurs for a third time since the start of the game 
        #       (not necessarily in succession): Declare draw
        
        # NOT IMPLEMENTED (save space)

        # 5.    The players have had their 360th turn without a winner being declared:
        #       Declare a draw.

        elif self.turn == 360:
            return 0

        return None


    def __moves_are_available(self, is_friend):
        """
        Return true if there are any moves or throws available to the player
        """
        if is_friend:
            return not (self.friend_throws == self.MAX_TOKENS and len(self.friends) == 0)
        else:
            return not (self.enemy_throws == self.MAX_TOKENS and len(self.enemies) == 0)

    def __player_is_invincible(self, is_friend):
        """
        Return true if player has at least one token that it is impossible for the other side
        to kill
        """
        if is_friend and self.enemy_throws < self.MAX_TOKENS:
            return False
        elif not is_friend and self.friend_throws < self.MAX_TOKENS:
            return False
        friend_tokens = self.__tokens_on_board(is_friend=True)
        enemy_tokens = self.__tokens_on_board(is_friend=False)
        if len(friend_tokens) == 3 and len(enemy_tokens) == 3:
            return False
        if is_friend:
            for friend_token in friend_tokens:
                if defeat_by_token(friend_token) not in enemy_tokens:
                    return True
                
            return False
        else:
            for enemy_token in enemy_tokens:
                if defeat_by_token(enemy_token) not in friend_tokens:
                    return True
            return False
        

    def __tokens_on_board(self, is_friend) -> set:
        """
        Return a set of all tokens currently on the board for a player
        """
        pieces = self.friends if is_friend else self.enemies
        tokens = set()
        for (token, _) in pieces:
            tokens.add(token)
        return tokens


    def update(self, enemy_move=None, friend_move=None):
        """
        Main update function.

        Takes a friend and enemy move from player and returns a new GameState with updates applied, 
        including any kills.
        """

        updated_pieces, new_throws = self.__updated_pieces(friend_move, enemy_move)
        return self.__update_kills(updated_pieces, new_throws)
        

    def __updated_pieces(self, friend_move, enemy_move):
        """
        Create a dict of type {location: [friendly piece], [enemy piece]} with the state of all 
        pieces after the move has been made but before battles have been resolved.
        """
        updated_pieces = {}
        friend_num_throws = 0
        enemy_num_throws = 0
        if friend_move is not None:
            if friend_move[0] == "THROW":
                self.__update_throw(updated_pieces, friend_move, True)
                friend_num_throws = 1
            else:
                self.__update_swing_slide(updated_pieces, friend_move, True)
        if enemy_move is not None:
            if enemy_move[0] == "THROW":
                self.__update_throw(updated_pieces, enemy_move, False)
                enemy_num_throws = 1
            else:
                self.__update_swing_slide(updated_pieces, enemy_move, False)
        return updated_pieces, (friend_num_throws, enemy_num_throws)

    
    def __update_swing_slide(self, updated_pieces, new_move, is_friend):
        """
        Update `updated_pieces` with the the new pieces for a particular side after the swing or 
        slide in `new_move` has been applied.

        `updated_pieces` should be a dict of the following type: 
            {location: [friendly piece], [enemy piece]}.

        `is_friend` flag used to determine if the friends or enemies are updated.
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
        Update `updated_pieces` with the the new pieces for a particular side after the throw in 
        `new_move` has been applied.

        `updated_pieces` should be a dict of the following type: 
            {location: [friendly piece], [enemy piece]}.

        `is_friend` flag used to determine if the friends or enemies are updated.
        """
        throws = 0
        (_, new_t, new_loc) = new_move
        existing_pieces = self.friends if is_friend else self.enemies
        GameState.__append_loc_piece(updated_pieces, (new_t, new_loc), is_friend)
        for existing_piece in existing_pieces:
            GameState.__append_loc_piece(updated_pieces, existing_piece, is_friend)
        # if is_friend:
            # self.friend_throws += 1
        

    
    def __append_loc_piece(board_pieces, piece, is_friend):
        """
        Add a new piece to piece's location in `board_pieces`.

        `board_pieces` should be a dict of the following type: 
            {location: [friendly piece], [enemy piece]}.

        `is_friend` flag used to determine if it is the friend or enemy piece list that needs to 
        be updated.
        """
        (_, loc) = piece
        try:
            board_pieces[loc][int(not is_friend)].append(piece)
        except:
            board_pieces[loc] = ([],[])
            board_pieces[loc][int(not is_friend)].append(piece)


    def __update_kills(self, updated_pieces, new_throws):
        """
        Run through all locations in `updated_pieces` and remove any pieces that lose the battles 
        at those locations.

        `board_pieces` should be a dict of the following type: 
            {location: [friendly piece], [enemy piece]}.

        Return a new GameState with all updated moves applied.
        """

        new_state = self.__copy_properties()
        new_state.friend_throws += new_throws[0]
        new_state.enemy_throws += new_throws[1]
        new_state.turn += 1
        for loc, (fr_pieces, en_pieces) in updated_pieces.items():
            if len(fr_pieces) + len(en_pieces) > 1:

                condemmed = {defeat_token(t) for (t, _) in fr_pieces + en_pieces}

                new_state.friends += [(t, loc) for (t, _) in fr_pieces if t not in condemmed]
                new_state.enemies += [(t, loc) for (t, _) in en_pieces if t not in condemmed]
            else:
                new_state.friends += fr_pieces
                new_state.enemies += en_pieces

        return new_state

    
    def next_transitions(self):
        """
        Return all possible friend enemy move combinations from the current `GameState`

        """

        all_moves = list(product(self.next_friend_transitions(), self.next_enemy_transitions()))
        return all_moves
    
    def next_friend_transitions(self):
        # return GameState.next_all_moves_for_side(self.friends, self.friend_throws, self.is_upper)
        return self.__next_moves_for_side(True) + self.__next_throws_for_side(True)

    
    def next_enemy_transitions(self):
        # return GameState.next_all_moves_for_side(self.enemies, self.enemy_throws, not self.is_upper)
        return self.__next_moves_for_side(False) + self.__next_throws_for_side(False)
    
    def next_friend_moves(self):
        return self.__next_moves_for_side(is_friend=True)

    def next_enemy_moves(self):
        return self.__next_moves_for_side(is_friend=False)

    def next_friend_throws(self):
        return self.__next_throws_for_side(is_friend=True)

    def next_enemy_throws(self):
        return self.__next_throws_for_side(is_friend=False)

    
    def __next_moves_for_side(self, is_friend):
        pieces = self.friends if is_friend else self.enemies
        moves = []
        for (_, loc) in pieces:
            slide_moves = GameState.__slide_transitions(loc)
            swing_moves = GameState.__swing_transitions(pieces, loc, slide_moves)
            moves += slide_moves + swing_moves
        return moves

    def __next_throws_for_side(self, is_friend):
        # pieces = self.friends if is_friend else self.enemies
        num_tokens_waiting = self.friend_throws if is_friend else self.enemy_throws

        if num_tokens_waiting < GameState.MAX_TOKENS:
            return self.__throw_transitions(num_tokens_waiting, is_friend)
        else:
            return []

    # def next_all_moves_for_side(pieces, num_tokens_waiting, is_upper):

    #     if num_tokens_waiting < GameState.MAX_TOKENS:
    #         moves = GameState.__throw_moves(num_tokens_waiting, is_upper)
    #     else:
    #         moves = []

    #     for (_, loc) in pieces:
    #         slide_moves = GameState.__slide_moves(loc)
    #         swing_moves = GameState.__swing_moves(pieces, loc, slide_moves)
    #         moves += slide_moves + swing_moves
    #     return moves


    def __throw_transitions(self, num_tokens_waiting, is_friend):
        """
        All possible throw moves

        Return tuple: ("THROW", token, location)
        """
        is_upper = self.is_upper if is_friend else not self.is_upper
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


    def __slide_transitions(loc):
        """
        All possible slide moves for a single location

        Return tuple: ("SLIDE", slide from location, slide to location)
        """

        result = []
        for d_loc in GameState.slide_options:
            if GameState.board.is_legal_location(loc_add(d_loc, loc)):
                result.append(("SLIDE", loc, loc_add(d_loc, loc)))
        

        return result


    def __swing_transitions(pieces, curr_loc, slide_moves):
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
                for (_, _, swing_loc) in GameState.__slide_transitions(other_loc):
                    if GameState.board.is_legal_location(swing_loc) \
                        and swing_loc not in pivots \
                        and swing_loc != curr_loc:
                        # swing destination must be a valid location, 
                        # somewhere that the piece could not slide to, 
                        # and cannot be the original piece location.
                        result.append(("SWING", curr_loc, swing_loc))

        return result


    def __hash__(self):
        return (str(self.friends) + str(self.enemies)).__hash__()


    def __str__(self):
        return "Upper: {}\nLower: {}".format(self.friends.__str__(), self.enemies.__str__())


    def __repr__(self):
        return self.__str__()


    def __eq__(self, other):
        return self.hash == other.hash
