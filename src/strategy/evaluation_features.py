import numpy as np
from math import log
from state.game_state import GameState
from state.token import defeats, defeat_token
from state.location import distance


class EvaluationFeatures:
    """
    Evaluation features used in machine learning process
    """

    middle_coords = {(0,0), (0,-1), (1,-1), (1,0), (0,1), (-1,1), (-1,0)}
    semi_middle_coords = {
        (2,-2), (2,-1), (2,0),
        (1,-2), (1,1),
        (0,-2), (0,2),
        (-1,-1), (-1,2),
        (-2,0), (-2,1), (-2,2)
    }
    outer_coords = {
        (4, -4), (4, -3), (4, -2), (4, -1), (4, 0),  # top edge
        (0, -4), (1, -4), (2, -4), (3, -4),  # left top edge
        (1, 3), (2, 2), (3, 1),  # top right edge
        (-4, 0), (-4, 1), (-4, 2), (-4, 3), (-4, 4),  # bottom edge
        (-1, -3), (-2, -2), (-3, -1),  # bottom left edge
        (0, 4), (-1, 4), (-2, 4), (-3, 4),  # bottom right edge
    }
    
    def __init__(self):
        self.throw_diff = 0
        self.death_diff = 0

        self.friend_has_kill_from_throw = 0  # bool
        self.friend_count_kill_from_throw = 0
        self.friend_has_kill_from_non_throw = 0  # bool
        self.friend_count_kill_from_non_throw = 0
        self.friend_has_stack = 0  # bool
        self.friend_dist_to_nearest_kill = 0
        self.friend_dist_to_all_kills = 0
        self.friend_has_invicible = 0
        self.friend_count_mid = 0


        self.enemy_has_kill_from_throw = 0  # bool
        self.enemy_count_kill_from_throw = 0
        self.enemy_has_kill_from_non_throw = 0  # bool
        self.enemy_count_kill_from_non_throw = 0
        self.enemy_has_stack = 0  # bool
        self.enemy_dist_to_nearest_kill = 0
        self.enemy_dist_to_all_kills = 0
        self.enemy_has_invicible = 0
        self.enemy_count_mid = 0


    def calculate_features(self, game_state):
        # battles
        self.throw_diff = game_state.friend_throws - game_state.enemy_throws
        self.death_diff = self.__count_kills(game_state, True) - self.__count_kills(game_state, False)

        # actions
        self.friend_has_kill_from_throw, self.friend_count_kill_from_throw = self.__kill_from_throw_values(game_state, True)
        self.friend_has_kill_from_non_throw, self.friend_count_kill_from_non_throw = self.__kill_from_non_throw_values(game_state, True)
        
        self.enemy_has_kill_from_throw, self.enemy_count_kill_from_throw = self.__kill_from_throw_values(game_state, False)
        self.enemy_has_kill_from_non_throw, self.enemy_count_kill_from_non_throw = self.__kill_from_non_throw_values(game_state, False)

        # locations
        self.friend_has_stack = self.__has_stack(game_state, True)
        self.friend_dist_to_nearest_kill, self.friend_dist_to_all_kills = self.__nearest_kill(game_state, True)
        self.friend_count_mid = log(self.__count_by_coords(game_state, True) + 1)
        
        self.enemy_has_stack = self.__has_stack(game_state, False)
        self.enemy_dist_to_nearest_kill, self.enemy_dist_to_all_kills = self.__nearest_kill(game_state, False)
        self.enemy_count_mid = log(self.__count_by_coords(game_state, False) + 1)
    
        # resources
        self.friend_has_invicible, self.enemy_has_invicible = self.__has_invincible(game_state, True)

    def to_vector(self):
        result = [0] * len(self.__dict__)
        for i, key in enumerate(self.__dict__):
            result[i] = self.__dict__[key]
        return result

    def __count_peices(self, game_state, is_friend):
        reference = game_state.friends if is_friend else game_state.enemies
        return sum([len(v) for v in reference.values()])

    def __count_kills(self, game_state, is_friend):
        throw_count = game_state.friend_throws if is_friend else game_state.enemy_throws
        return throw_count - self.__count_peices(game_state, is_friend)

    def __kill_from_throw_values(self, game_state, is_friend):
        throw_options = self.__calc_throw_options(game_state, is_friend)

        throw_kills_values = self.__count_kills_from_throw(game_state, is_friend, throw_options)
        has_throw_kill_option = 1 if (throw_kills_values > 0) else 0

        return has_throw_kill_option, throw_kills_values  # bool to indicate throw kill option exists, and amount of options

    def __count_kills_from_throw(self, game_state, is_friend, throw_options):
        if len(throw_options) == 0: return 0  # no throw options
        
        count = 0
        other_reference = game_state.enemies if is_friend else game_state.friends
        
        for loc in other_reference:  # enemy locations
            if loc in throw_options:  # that you can throw a token on to
                count += 1
        
        return count

    def __calc_throw_options(self, game_state, is_friend):
        throw_count = game_state.friend_throws if is_friend else game_state.enemy_throws
        if throw_count >= GameState.MAX_THROWS:
            return []  # no available throws

        upper = (game_state.is_upper and is_friend) or ((not game_state.is_upper) and (not is_friend))
        return GameState.board.get_throw_options(upper, throw_count)

    def __kill_from_non_throw_values(self, game_state, is_friend):
        reference = game_state.friends if is_friend else game_state.enemies
        other_reference = game_state.enemies if is_friend else game_state.friends

        count = 0

        for f_loc in reference:  # friendly pieces
            slide_options = GameState.board.get_slide_options(f_loc)
            for slide_loc in slide_options:
                if slide_loc in other_reference:  # there is an opponent that can be slid on to
                    if defeats(reference[f_loc][0], other_reference[slide_loc][0]):  # 0 index is the token on the location
                        count += 1

            for pivot in slide_options:
                if pivot in reference:
                    swing_options = GameState.board.get_swing_options(f_loc, pivot)

                    for swing_loc in swing_options:
                        if swing_loc in other_reference:
                            if defeats(reference[f_loc][0], other_reference[swing_loc][0]):  # 0 index is the token on the location
                                count += 1
            
        has_non_throw_kill = 1 if (count > 0) else 0

        return has_non_throw_kill, count

    def __has_invincible(self, game_state, is_friend):
        reference = game_state.friends if is_friend else game_state.enemies
        other_reference = game_state.enemies if is_friend else game_state.friends

        f_types = set([t for list_t in reference.values() for t in list_t])
        e_types = set([t for list_t in other_reference.values() for t in list_t])
        
        f_defeats = set([defeat_token(t) for t in f_types])
        e_defeats = set([defeat_token(t) for t in e_types])

        f_remain = f_types - e_defeats
        e_remain = e_types - f_defeats

        return (len(f_remain) > 0), (len(e_remain) > 0)

    def __has_stack(self, game_state, is_friend):
        reference = game_state.friends if is_friend else game_state.enemies

        for v in reference.values():
            if len(v) > 1:
                return True
        
        return False

    def __nearest_kill(self, game_state, is_friend):
        reference = game_state.friends if is_friend else game_state.enemies
        other_reference = game_state.enemies if is_friend else game_state.friends

        kill_distances = []

        for f_loc in reference:
            instance_distance = False  # high base value
            for e_loc in other_reference:
                if defeats(reference[f_loc][0], other_reference[e_loc][0]):
                    kill_dist = distance(f_loc, e_loc)
                    if not instance_distance:
                        instance_distance = kill_dist
                    elif kill_dist < instance_distance:
                        instance_distance = kill_dist
            
            if instance_distance:
                kill_distances.append(instance_distance)
        
        if not kill_distances: return 0, 0

        return min(kill_distances), sum(kill_distances)

    def __count_by_coords(self, game_state, is_friend):
        reference = game_state.friends if is_friend else game_state.enemies

        mids, semi_mids, semi_out, outers = 0, 0, 0, 0

        for loc in reference:
            if loc in EvaluationFeatures.middle_coords:
                mids += 1
            elif loc in EvaluationFeatures.semi_middle_coords:
                semi_mids += 1
            elif loc in EvaluationFeatures.outer_coords:
                outers += 1
            else:
                semi_out += 1

        return mids + semi_mids
