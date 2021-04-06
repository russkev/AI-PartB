"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

def defeat_token(t):
        if t == 'r':
            return 's'
        elif t == 'p':
            return 'r'
        else:
            return 'p'

def defeat_by_token(t):
    if t == 'r':
        return 'p'
    elif t == 'p':
        return 's'
    else:
        return 'r'

def defeats(a_t, b_t):
    return (a_t == "r" and b_t == "s") \
        or (a_t == "p" and b_t == "r") \
        or (a_t == "s" and b_t == "p")
