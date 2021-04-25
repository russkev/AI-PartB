"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from state.game_state import GameState


class Node(GameState):
    def __init__(self, other: GameState):
        """
        Initialise with a GameState.
        Set is_friend parameter to false since our starting position (root) will always be the case 
        where enemy has just moved and we're choosing where to move. This is making the (incorrect)
        assumption that moves are sequential instead of simultaneous.
        """
        super().__init__()
        self.is_upper = other.is_upper
        self.friends = other.friends
        self.enemies = other.enemies
        self.turn = other.turn
        self.friend_throws = other.friend_throws
        self.enemy_throws = other.enemy_throws

        self.action = None  # Action made to make this move
        self.parent = None
        # self.children = set()
        self.children = []
        self.is_friend = True
        # self.is_visited = False
        self.is_fully_expanded = False

        self.q_value = 0
        self.num_visits = 0

    def unvisited_children(self):
        """
        Return a list of children who have not been the root of a rollout yet.
        """
        unvisited = []
        for child in self.children:
            # if not child.is_visited:
            if child.num_visits == 0:
                unvisited.append(child)
        return unvisited

    def __repr__(self):
        return f"{self.q_value}/{self.num_visits}"
