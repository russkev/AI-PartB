"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from state.location import distance
from state.game_state import GameState



class Node(GameState):
    MAX_DISTANCE = 2

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
        # self.is_friend = False
        self.is_visited = False
        self.is_fully_expanded = False

        self.q_value = 0
        self.num_visits = 0

        self.friend_transitions = []
        self.enemy_transitions = []
        self.regret = []
        self.matrix = [[]]
        self.i = 0
        self.j = 0

    def init_transitions(self):
        self.friend_transitions = self.next_friend_moves()
        self.friend_transitions += self.__prune_throws(
            self.next_friend_throws(), True)

        self.enemy_transitions = self.next_enemy_moves()
        self.enemy_transitions += self.__prune_throws(
            self.next_enemy_throws(), False)

        self.init_matrix()

    def __prune_throws(self, throws, is_friend):
        throw_enemies = self.enemies if is_friend else self.friends
        pruned_throws = []

        if is_friend:
            tokens_remaining = self.friend_throws
        else:
            tokens_remaining = self.enemy_throws
        if (is_friend and self.is_upper) or (not is_friend and not self.is_upper):
            pruned_throws += Node.__append_throws_distant(
                throws, tokens_remaining, throw_enemies, True)
        else:
            pruned_throws += Node.__append_throws_distant(
                throws, tokens_remaining, throw_enemies, False)

        if len(pruned_throws) == 0:
            # There are opposing tokens less than MAX_DISTANCE away
            for throw in throws:
                (_, _, throw_loc) = throw
                for (_, enemy_loc) in throw_enemies:
                    if distance(throw_loc, enemy_loc) <= Node.MAX_DISTANCE:
                        pruned_throws.append(throw)
                        break

        return pruned_throws

    @staticmethod
    def __append_throws_distant(throws, tokens_remaining, throw_enemies, is_upper):
        # farthest_r = -4 if is_upper else 4
        farthest_r = GameState.farthest_r(tokens_remaining, is_upper)
        nearest_throw_enemy_r = -4 if is_upper else 4
        pruned_throws = []
        for (_, (r, _)) in throw_enemies:
            if r > nearest_throw_enemy_r:
                nearest_throw_enemy_r = r
        if abs(farthest_r - nearest_throw_enemy_r) > Node.MAX_DISTANCE:
            for throw in throws:
                (_, _, (throw_r, _)) = throw
                if throw_r == farthest_r:
                    pruned_throws.append(throw)
        return pruned_throws

    def init_matrix(self):
        self.matrix = [
            [None for _ in range(len(self.enemy_transitions))]
            for _ in range(len(self.friend_transitions))]

        for _ in range(len(self.friend_transitions)):
            self.regret.append(0)

    def has_unvisited_children(self):
        if self.is_fully_expanded:
            return False
        else:
            for friend_index in range(len(self.friend_transitions)):
                for enemy_index in range(len(self.enemy_transitions)):
                    if self.matrix[friend_index][enemy_index] is None:
                        return True
            self.is_fully_expanded = True
            return False

    def __repr__(self):
        return f"{self.q_value}/{self.num_visits}"
