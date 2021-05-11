"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""
import numpy as np
import random

def sigmoid(x, a, b):
    return 1/(1 + np.exp(-a*x+b))

def boolean_from_probability(prob):
    return random.random() < prob