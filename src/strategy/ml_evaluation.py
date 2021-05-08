import numpy as np
from math import exp
from state import game_state
from strategy.evaluation_features import EvaluationFeatures


model_coefs = np.array([1, 1, 0, 0, 0, 0, 10])

# new_model_coefs = np.array([-1000, -100, 10, -1000, 10000, 100,100])
new_model_coefs = np.array([-0.5204512950203091, 0.26902233485838695, 0.04382228393878375, -0.14619884216819995, 0.5410969366753706, -0.28169461333049095, -1.2154954982003174])

e = EvaluationFeatures()

def evaluate(game_state, use_new_model_fit=True):
    e.calculate_features(game_state)

    if use_new_model_fit: return np.array(e.to_vector()).dot(new_model_coefs)

    return np.array(e.to_vector()).dot(new_model_coefs)
