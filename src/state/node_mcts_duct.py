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


        self.parent = None
        self.is_fully_expanded = False

        self.friend_transitions = []
        self.enemy_transitions = []
        self.matrix = [[]]  # Friend is row player, enemy is column player

        self.q_value = 0
        self.num_visits = 0


    def unvisited_children(self):
        """
        Return a list of children who have not been the root of a rollout yet.
        """

        unvisited = []
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                child: Node = self.matrix[i][j]
                if child.num_visits == 0:
                    unvisited.append(child)
        return unvisited

    def update_node(self, opponent_action, player_action, parent = None):
        """
        If the new state already exists in the matrix, return that, otherwise return a fresh node

        the parent argument is for updating a simulation inside MCTS. If this is used as the actual
        update method, parent should stay default None
        """
        friend_index = enemy_index = -1
        if parent is None:
            # Find new root, parent will be None in this case
            for i, friend_transition in enumerate(self.friend_transitions):
                if friend_transition == player_action:
                    friend_index = i
            
            for j, enemy_transition in enumerate(self.enemy_transitions):
                if enemy_transition == opponent_action:
                    enemy_index = j

        if friend_index > -1 and enemy_index > -1:
            updated_node: Node = self.matrix[friend_index][enemy_index]
            updated_node.parent = None
            return updated_node
        else:
            updated_node = Node(self.update(opponent_action, player_action))
            if parent is not None:
                updated_node.parent = parent
            return updated_node



    def __repr__(self):
        """
        Display the value / num_visits. This is nice and small so fits into a matrix.
        """
        return f"{self.q_value}/{self.num_visits}"
