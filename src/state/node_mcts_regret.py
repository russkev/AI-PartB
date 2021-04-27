"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from state.game_state import GameState


class Node(GameState):
    def __init__(
        self,
        other: GameState,
        parent=None,
        is_fully_expanded=False,
        friend_transitions=None,
        enemy_transitions=None,
        matrix=None,
        regret=None,
        q_value=0,
        num_visits=0,
        i=-1,
        j=-1,
    ):
        """
        Initialise with a GameState.
        Set is_friend parameter to false since our starting position (root) will always be the case 
        where enemy has just moved and we're choosing where to move. This is making the (incorrect)
        assumption that moves are sequential instead of simultaneous.
        """
        super().__init__(
            is_upper=other.is_upper,
            friends=other.friends,
            enemies=other.enemies,
            turn=other.turn,
            friend_throws=other.friend_throws,
            enemy_throws=other.enemy_throws,
        )

        self.parent = parent
        self.is_fully_expanded = is_fully_expanded

        if friend_transitions is None:
            self.friend_transitions = []
        else:
            self.friend_transitions = friend_transitions

        if enemy_transitions is None:
            self.enemy_transitions = []
        else:
            self.enemy_transitions = enemy_transitions

        if matrix is None:
            self.matrix = [[]]  # Friend is row player, enemy is column player
        else:
            self.matrix = matrix
        if regret is None:
            self.regret = []
        else:
            self.regret = regret
        self.i = i
        self.j = j


        self.q_value = q_value
        self.num_visits = num_visits

    def copy_node_state(self) -> "Node":
        return Node(super().copy())

    def unvisited_children(self):
        """
        Return a list of children who have not been the root of a rollout yet.
        """

        unvisited = []
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                if self.matrix[i][j].num_visits == 0:
                    unvisited.append(self.matrix[i][j])
        return unvisited

    def make_updated_node(self, friend_transition, enemy_transition, parent=None, i=-1, j=-1):
        """
        If the new state already exists in the matrix, return that, otherwise return a fresh node

        the parent argument is for updating a simulation inside MCTS. If this is used as the actual
        update method, parent should stay default None
        """
        friend_index = enemy_index = -1
        if parent is None:
            # Find new root, parent will be None in this case
            for i, curr_friend_transition in enumerate(self.friend_transitions):
                if curr_friend_transition == friend_transition:
                    friend_index = i

            for j, curr_enemy_transition in enumerate(self.enemy_transitions):
                if curr_enemy_transition == enemy_transition:
                    enemy_index = j

        if friend_index > -1 and enemy_index > -1:
            # Correct child node found
            updated_node: Node = self.matrix[friend_index][enemy_index]
            updated_node.parent = None
            return updated_node
        else:
            # New node needs to be created
            updated_node = self.copy_node_state()
            updated_node.update(friend_transition, enemy_transition)
            updated_node.i = i
            updated_node.j = j
            # updated_node.__init_arrays()
            if parent is not None:
                updated_node.parent = parent
            return updated_node

    def __repr__(self):
        """
        Display the value / num_visits. This is nice and small so fits into a matrix.
        """
        return f"{self.q_value}/{self.num_visits}"