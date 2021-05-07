"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

import sys, subprocess

DEFAULT_ITERATIONS = 1

def main():

    if len(sys.argv) < 2:
        print("Not enough arguments supplied, exiting")
        sys.exit(1)
    
    upper = sys.argv[1]
    lower = sys.argv[2]

    if len(sys.argv) >= 4:
        iterations = int(sys.argv[3])
    else:
        iterations = DEFAULT_ITERATIONS

    num_upper_wins = 0
    num_draws = 0

    for i in range(iterations):
        game_result = subprocess.run(
            ["python3", "-m", "referee", upper, lower, "-v" "0"],
            capture_output = True,
            encoding = "utf-8"
            )

        result = game_result.stdout[2:-1]
        winning_algorithm = ""
        if result == "winner: upper":
            winning_algorithm = upper
            num_upper_wins += 1
        elif result == "winner: lower":
            winning_algorithm = lower
        else:
            num_draws += 1
        print("Game: {}, result: {}, {}".format(i, result, winning_algorithm))
    
    upper_win_ratio = num_upper_wins / iterations
    num_lower_wins = iterations - num_upper_wins - num_draws
    lower_win_ratio = num_lower_wins / iterations
    draw_ratio = num_draws / iterations

    print(f"\nUpper wins: {num_upper_wins}/{iterations}, with ratio: {upper_win_ratio:.2f}")
    print(f"Lower wins: {num_lower_wins}/{iterations}, with ratio: {lower_win_ratio:.2f}")
    print(f"Draws: {num_draws}/{iterations}, with ratio: {draw_ratio:.2f}")



    # parser = argparse.ArgumentParser(
    #     description = "Process player arguments")
    
    # positionals = parser.add_argument_group(title="players")
    
    # for num, col in enumerate(["Upper", "Lower"], 1):
    #     positionals.add_argument(f"player{num}_loc", metavar=col)

    # # parser.add_argument("player 1", type = str)
    # # parser.add_argument("player 2", type = str)
    # parser.parse_args()

    return

