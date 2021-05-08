from os import listdir
import pandas as pd


def prepare_ml_data_set(keep_draws = True):
    game_files = sorted([f for f in listdir() if f[-4:] =='.csv' and f != 'game_outcomes.csv'])

    header = ['throw_diff',
        'death_diff',
        'nearest_kill_dist_diff',
        'total_kill_dist_diff',
        'count_mid_diff',
        'stack_diff',
        'invincible_diff',
        'kill_from_throw_count_diff',
        'kill_from_non_throw_count_diff',
        'outcome']
   
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