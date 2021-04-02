"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""


class Board:

    def __init__(self):
        self.locations = Board.__locations()

    @staticmethod
    def __locations():
        """
        Creates a list of tuples which are the co-ordinates on the game board.

        returns : Set
            {(row, column), ...}

        """

        # Inspired by util print function
        coord_range = range(-4, 5)
        return {(r, q) for r in coord_range for q in coord_range if -r-q in coord_range}

    def is_legal_location(self, loc):
        return loc in self.locations
        # if loc not in self.locations:
        #     return False

        # return True


    # def friends(self):
    #     """
    #     Return a list of all friendly pieces on the board
    #     """
    #     # friends = []
    #     # for loc, ((rs, ps, ss), *_) in self.locations.items():
    #     #     friends += rs * [loc] + ps * [loc] + ss * [loc]
    #     # return friends
    #     return [piece for pieces in self.locations.values() for piece in pieces]
    
    # def add_friend(self, piece):
    #     (t, loc) = piece
    #     self.locations[loc][0].append((t, loc))