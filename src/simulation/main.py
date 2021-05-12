"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088

Simulation program

Takes two agents and an optional simulation amount as inputs from the command line.
Runs simulated games the specified number of times.
"""

import sys, subprocess
import csv
from subprocess import PIPE

DEFAULT_ITERATIONS = 30

def main():

    if len(sys.argv) < 2:
        print("Not enough arguments supplied, exiting")
        sys.exit(1)
    
    upper = sys.argv[1]
    lower = sys.argv[2]
    print(upper)
    print(lower)

    if len(sys.argv) >= 4:
        iterations = int(sys.argv[3])
    else:
        iterations = DEFAULT_ITERATIONS

    num_upper_wins = 0
    num_draws = 0

    game_outcome = 0.5

    for i in range(iterations):
        # Run the simulation
        game_result = subprocess.run(
            ["python3", "-m", "referee", upper, lower, "-v" "0"],
            stdout=PIPE,
            encoding = "utf-8"
            )

        result = game_result.stdout[2:-1]
        winning_algorithm = ""

        # Print results for this simulations
        if result == "winner: upper":
            winning_algorithm = upper
            num_upper_wins += 1
            game_outcome = 1
        elif result == "winner: lower":
            winning_algorithm = lower
            game_outcome = 0
        else:
            num_draws += 1
            game_outcome = 0.5
        
        with open('game_logs/game_outcomes.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([game_outcome])

        print(f"Game: {i}, result: {result}, {winning_algorithm}")
    
    upper_win_ratio = num_upper_wins / iterations
    num_lower_wins = iterations - num_upper_wins - num_draws
    lower_win_ratio = num_lower_wins / iterations
    draw_ratio = num_draws / iterations

    # Print the final results
    print(f"\nUpper wins: {num_upper_wins}/{iterations}, with ratio: {upper_win_ratio:.2f}")
    print(f"Lower wins: {num_lower_wins}/{iterations}, with ratio: {lower_win_ratio:.2f}")
    print(f"Draws: {num_draws}/{iterations}, with ratio: {draw_ratio:.2f}")

    return