from os import listdir
import pandas as pd


def prepare_ml_data_set(keep_draws = True):
    game_files = sorted([f for f in listdir() if f[-4:] =='.csv' and f != 'game_outcomes.csv'])

    header = [
        'friend_throws','friend_deaths','friend_has_kill_from_throw','friend_count_kill_from_throw','friend_has_kill_from_non_throw'
        ,'friend_count_kill_from_non_throw','friend_has_stack','friend_dist_to_nearest_kill','friend_dist_to_all_kills','friend_has_invicible'
        ,'friend_count_mid','friend_count_semi_mid','friend_count_semi_outer','friend_count_outer','enemy_throws','enemy_deaths','enemy_has_kill_from_throw'
        ,'enemy_count_kill_from_throw','enemy_has_kill_from_non_throw','enemy_count_kill_from_non_throw','enemy_has_stack','enemy_dist_to_nearest_kill'
        ,'enemy_dist_to_all_kills','enemy_has_invicible','enemy_count_mid','enemy_count_semi_mid','enemy_count_semi_outer','enemy_count_outer', 'outcome'
    ]

    outcomes = pd.read_csv('game_outcomes.csv',header=None)
    df = pd.DataFrame(columns=header)

    for i, f in enumerate(game_files):
        df_new = pd.read_csv(f, header=None)
        df_new[len(header)-1] = outcomes[0][i]
        df_new.columns = header

        df = pd.concat([df, df_new], ignore_index=True)

    if keep_draws:
        return df
    
    return df[df['outcome'] != 0.5]