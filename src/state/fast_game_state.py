from state.board import Board


class GameState:

    MAX_TOKENS = 9
    MAX_TURNS = 360
    slide_options = [(r, q) for r in [-1, 0, 1] for q in [-1, 0, 1] if (abs(r + q) < 2) and (r != 0 or q != 0)]
    board = Board(slide_options)

    def __init__(self):
        pass



