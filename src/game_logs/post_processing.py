from os import listdir
import numpy as np
import pandas as pd


def prepare_ml_data_set():
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
        'score']
   
    outcomes = pd.read_csv('game_outcomes.csv',header=None)
    df = pd.DataFrame(columns=header)

    for i, f in enumerate(game_files):
        f = game_files[0]
        df_new = pd.read_csv(f, header=None)
        df_new[len(header)-1] = 0
        df_new.columns = header

        turn = list(range(len(df_new.score)))
        death_diff = np.cumsum(df_new['death_diff'])
        outcome = outcomes[0][i]

        df_new['score'] = 50 * outcome - 5 * death_diff - turn[::-1]

        df = pd.concat([df, df_new], ignore_index=True)
    
    return df