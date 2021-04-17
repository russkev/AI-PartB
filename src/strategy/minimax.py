import math


def minimax(node, depth, maximising_player):

    if depth == 0:
        return node
    
    if maximising_player:
        value = -math.inf
        for move in node[1]:
            value = max(value, minimax(move, depth - 1, False))
        return value

    else:
        value = math.inf
        for move in node[1]:
            value = min(value, minimax(move, depth - 1, False))
        return value


# def add_turns(game_state, friend_turns, enemy_turn):



# def ropasci_minimax(game_state, depth, maximising_player)
#         if depth == 0:
#         return game_state
    
#     if maximising_player:
#         value = -math.inf
#         for move in self.game_state.next_moves():
#             value = max(value, minimax(move, depth - 1, False))
#         return value

#     else:
#         value = math.inf
#         for move in self.game_state.next_moves():
#             value = min(value, minimax(move, depth - 1, False))
#         return value


# def evaluation_function(game_state):
#     """
#     Takes a game state and assigns an estimated utility value.
#     """
#     return len(game_state.upper)


if __name__ == '__main__':
    g = (None, ((None, (1, 3)), (None, (2, 4))))

    # game_state = GameState()        
    # game_state.is_upper = "upper"
    # self.game_state = self.game_state.update(opponent_action, player_action)


    # game_state.next_friend_moves


# function minimax(node, depth, maximizingPlayer) is
#     if depth = 0 or node is a terminal node then
#         return the heuristic value of node
#     if maximizingPlayer then
#         value := −∞
#         for each child of node do
#             value := max(value, minimax(child, depth − 1, FALSE))
#         return value
#     else (* minimizing player *)
#         value := +∞
#         for each child of node do
#             value := min(value, minimax(child, depth − 1, TRUE))
#         return value