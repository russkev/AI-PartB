from os import listdir
import numpy as np
import pandas as pd


def prepare_ml_data_set():
    game_files = sorted([f for f in listdir() if f[-4:] =='.csv' and f != 'game_outcomes.csv'])

    header = ['death_diff',
        'nearest_kill_dist_diff',
        'has_mid_diff',
        'stack_diff',
        'invincible_diff',
        'has_kill_from_throw_diff',
        'has_kill_from_non_throw_diff',
        'score']
   
    outcomes = pd.read_csv('game_outcomes.csv',header=None)
    df = pd.DataFrame(columns=header)
    discount_factor = 0.7

    for i, f in enumerate(game_files):
        f = game_files[0]
        df_new = pd.read_csv(f, header=None)
        df_new[len(header)-1] = 0
        df_new.columns = header

        turn = list(range(len(df_new.score)))
        outcome = outcomes[0][i]
        factor = [discount_factor**t for t in turn[::-1]]

        df_new['score'] = ((100 * outcome) - (5 * df_new['death_diff']) - (turn[::-1])) * factor

        df = pd.concat([df, df_new], ignore_index=True)
    
    return df