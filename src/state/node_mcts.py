"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from state.game_state_fast import GameState


class Node(GameState):
    def __init__(
        self, 
        other: GameState, 
        action=None, 
        parent=None, 
        children=None, 
        is_friend=True, 
        is_fully_expanded=False,
        q_value=0,
        num_visits=0
    ):
        """
        Initialise with a GameState.
        Set is_friend parameter to false since our starting position (root) will always be the case 
        where enemy has just moved and we're choosing where to move. This is making the (incorrect)
        assumption that moves are sequential instead of simultaneous.
        """
        super().__init__(
            is_upper=other.is_upper, 
            turn=other.turn,
            friend_throws = other.friend_throws,
            enemy_throws = other.enemy_throws,
            friends = other.friends,
            enemies = other.enemies,
        )

        # Extra attributes
        self.action = action  # Action made to make this move
        self.parent = parent
        if children is None:
            self.children = []
        else:
            self.children = children
        self.is_friend = is_friend
        self.is_fully_expanded = is_fully_expanded

        # Statistics
        self.q_value = q_value
        self.num_visits = num_visits

    def copy_node_state(self) -> "Node":
        return Node(super().copy())


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

    def update_node(self, player_action, opponent_action, parent=None):
        """
        If the new move is in the tree, return that instead of creating a new root
        """
        for pl_child in self.children:
            if pl_child.action == player_action:
                for op_child in pl_child.children:
                    if op_child.action == opponent_action:
                        op_child.parent = parent
                        return op_child
        
        # New move has not been found, make a new one.
        new_root = self.copy_node_state()
        new_root.update(player_action, opponent_action)
        return new_root
