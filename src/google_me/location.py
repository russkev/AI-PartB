"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

# Distance algorithm from Red Blob Games
# https://www.redblobgames.com/grids/hexagons/
def distance(loc_a, loc_b):
    return (
        abs(loc_a[0] - loc_b[0]) + 
        abs(loc_a[1] - loc_b[1]) + 
        abs(-loc_a[1] - loc_a[0] + loc_b[1] + loc_b[0])) / 2

def loc_add(loc_a, loc_b):
    return (loc_a[0] + loc_b[0], loc_a[1] + loc_b[1])

def loc_lower_than(loc, row):
    return loc[0] < row

def loc_higher_than(loc, row):
    return loc[0] > row
