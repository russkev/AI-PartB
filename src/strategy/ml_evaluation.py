import numpy as np
from math import exp
from state import game_state
from strategy.evaluation_features import EvaluationFeatures

logistic_coefs = np.array([-1.53957804e+00, -3.95491844e-01, -6.25767776e-02,
        -3.51937374e-01,  1.77480062e-03,  7.23573021e-02,
        -3.80452455e+00,  6.41608187e-01,  2.56296897e-01,
         6.53878786e-01,  4.09788485e-01,  1.17902930e-02,
        -3.66490241e-01,  6.52283929e-02, -7.85988248e-01,
         3.52134222e+00, -4.71307473e-01, -1.11872757e+00,
        -1.50879422e+00, -6.08569972e-01])

logistic_inter = 0.4032599

e = EvaluationFeatures()

def evaluate(game_state):
    e.calculate_features(game_state)

    raw = np.array(e.to_vector()).dot(logistic_coefs) + logistic_inter

    return 1 / (1 + exp(-raw))

