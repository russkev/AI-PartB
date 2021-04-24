"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from state.location import loc_add


class Board:

    def __init__(self, slide_options):
        self.locations = Board.__locations()
        self.slide_options = slide_options
        self.slide_lookup = self.__compute_slide()
        self.swing_lookup = self.__compute_swing()
        self.lower_throw_lookup = self.__compute_lower_throw()
        self.upper_throw_lookup = self.__compute_upper_throw()

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

    def get_slide_options(self, piece_location):
        return self.slide_lookup[piece_location]

    def get_swing_options(self, piece_location, ally_location):
        return self.swing_lookup[piece_location][ally_location]

    def get_throw_options(self, upper, throw_count):
        if upper:
            return self.upper_throw_lookup[throw_count]
        
        return self.lower_throw_lookup[throw_count]

    def __compute_slide(self):
        slide_lookup = {}

        for loc in self.locations:
            options = set()
            for d_loc in self.slide_options:    
                if self.is_legal_location(loc_add(d_loc, loc)):
                    options.add(loc_add(d_loc, loc))

            slide_lookup[loc] = options

        return slide_lookup

    def __compute_swing(self):
        swing_lookup = {}
        for loc in self.locations:
            pivots = self.slide_lookup[loc]

            swing_lookup[loc] = {}
            for ally_loc in pivots:  # where the ally peices can be
                swing_lookup[loc][ally_loc] = set()
                for swing_dest in self.slide_lookup[ally_loc]:  # all swing options
                    if (swing_dest != loc) and (swing_dest not in self.slide_lookup[loc]):
                        # can't swing to a tile that you can also slide to, and can't swing to original location
                        swing_lookup[loc][ally_loc].add(swing_dest)
        
        return swing_lookup

    def __compute_lower_throw(self):
        lower_throw_lookup = {}
        start_row = -4
        for throw in range(0,9):
            lower_throw_lookup[throw] = set()
            for (r, q) in self.locations:
                if r <= (start_row + throw):
                    lower_throw_lookup[throw].add((r, q))
        
        return lower_throw_lookup

    def __compute_upper_throw(self):
        upper_throw_lookup = {}
        start_row = 4
        for throw in range(0,9):
            upper_throw_lookup[throw] = set()
            for (r, q) in self.locations:
                if r >= (start_row - throw):
                    upper_throw_lookup[throw].add((r, q))
        
        return upper_throw_lookup


if __name__ == '__main__':
    slide_options = [(r, q) for r in [-1, 0, 1] for q in [-1, 0, 1] if (abs(r + q) < 2) and (r != 0 or q != 0)]
    b = Board(slide_options)

