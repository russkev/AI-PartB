# most of this file is a greedy implementation that can be moved to the strategy folder.

"""
COMP30024 Artificial Intelligence
Semester 1, 2021
Project Part B
David Peel 964682
Kevin Russell 1084088
"""

from state.game_state import GameState
from state.location import distance
from state.token import defeats

import time

import heapq
class Agent:

    def __init__(self, game_state):
        self.path = []
        self.path_states = []
        self.game: GameState = game_state
        self.round = 0


        self.state_count = 0
        self.iter_count = 0
        self.max_storage = 0
        self.max_depth = 0
        self.time_finding_best_state = 0
        self.time_generating_new_states = 0
        self.time_updating_states = 0

    def __calculate_h(state):
        """
        returns the heuristic value for the board state
        """
        score = 0
        for(lower_t, lower_r, lower_q) in state.lower:
            min_s = 10000
            for (upper_t, upper_r, upper_q, *_) in state.upper:
                if defeats(upper_t, lower_t):
                    curr_dist = distance(upper_r, upper_q, lower_r, lower_q)
                    if curr_dist < min_s:
                        min_s = curr_dist

            score += min_s/2 

        return score

    # def next_move(self, algorithm, doPrintExtra):

    #     if self.round == 0:
    #         self.start_time = time.time()
    #         if(algorithm == "bfs"):
    #             self.__bfs()
    #         elif(algorithm == "greedy"):
    #             self.__greedy()
    #         elif algorithm == "astar2":
    #             self.__a_star_2()
    #         elif algorithm == "astar3":
    #             self.__a_star_3()
    #         else:
    #             print("Error: Invalid algorithm: ", algorithm)
    #             exit()
    #         self.round += 1
         
    #     if self.round >= len(self.path):
    #         self.end_time = time.time()
    #         return False
        
    #     # Only care about upper state for printing
    #     for (t, r, q, move, prev_r, prev_q, *_) in self.path[self.round].moves:
    #         if move == "slide":
    #             print_slide(self.round, prev_r, prev_q, r, q)
    #         elif move == "swing":
    #             print_swing(self.round, prev_r, prev_q, r, q)
    #         else:
    #             print("Error: {} is an invalid move command".format(move))
    #     if doPrintExtra:
    #         Display.game_state(self.path[self.round])


    #     self.round += 1
    #     return True

    def __trace_path(self, state: GameState):
        """
        Upon a star success
        """
        while state is not None:
            self.path.append(state)
            state = state.parent
        self.path.reverse()
        return self.path

    def __a_star_2(self):
        h = []

        start_state = self.game
        h_cost = Agent.__calculate_h(start_state)
        start_state.costs = (h_cost, 0, h_cost)
        
        heapq.heappush(h, (start_state.costs[0], start_state))
        closed = set()

        while len(h) > 0:
            start_time = time.time()
            (_, explore_state) = heapq.heappop(h)
            end_time = time.time()
            self.time_finding_best_state += end_time - start_time

            closed.add(explore_state)

            self.iter_count+=1
            self.max_storage = max(len(h) + len(closed), self.max_storage)

            if explore_state.is_goal_state() or time.time() - self.start_time > 29:
                self.__trace_path(explore_state)
                return

            self.max_depth = max(explore_state.costs[1] + 1, self.max_depth)
            # if explore_state.is_goal_state():
            #     self.__trace_path(explore_state)
            #     return

            start_time = time.time()
            new_states = explore_state.next_states()
            end_time = time.time()
            self.time_generating_new_states += end_time - start_time

            for new_state in new_states:
                if new_state in closed:
                    continue

                self.state_count += 1
                new_state.parent = explore_state
                new_g_cost = explore_state.costs[1] + 1

                start_time = end_time
                new_h_cost = Agent.__calculate_h(new_state)
                end_time = time.time()
                self.time_updating_states += end_time-start_time

                new_f_cost = new_g_cost + new_h_cost
                new_state.costs = (new_f_cost, new_g_cost, new_h_cost)
                heapq.heappush(h, (new_state.costs[0], new_state))

        return

    def __a_star_3(self):

        start_state = self.game
        g_cost = 0
        h_cost = Agent.__calculate_h(start_state)
        f_cost = g_cost + h_cost
        start_state.costs=(f_cost, g_cost, h_cost)

        open_dict = {start_state: start_state}

        open_heap = []
        heapq.heappush(open_heap, start_state)
        
        closed_set = set()

        while len(open_heap) > 0:
            start_time = time.time()

            explore_state = heapq.heappop(open_heap)
            del open_dict[explore_state]

            end_time = time.time()
            self.time_finding_best_state += end_time - start_time

            closed_set.add(explore_state)

            self.iter_count+=1
            self.max_storage = max(len(open_dict) + len(closed_set), self.max_storage)

            if explore_state.is_goal_state() or time.time() - self.start_time > 29:
                self.__trace_path(explore_state)
                return
            
            self.max_depth = max(explore_state.costs[1] + 1, self.max_depth)


            start_time = time.time()
            (new_states, move_states) = explore_state.next_states()
            end_time = time.time()
            self.time_generating_new_states += end_time - start_time
            
            # for new_state in new_states:
            for i, new_state in enumerate(new_states):

                if new_state in closed_set:
                    continue
                new_g_cost = explore_state.costs[1] + 1

                # Only update `open` if it does not contain the new state
                # or the state it does contain has a higher g_cost
                if Agent.__cost_needs_updating(open_dict, new_state, new_g_cost):
                    self.state_count += 1
                    start_time = end_time
                    new_h_cost = Agent.__calculate_h(new_state)
                    end_time = time.time()
                    self.time_updating_states += end_time-start_time
                    # new_f_cost = new_g_cost + new_h_cost
                    new_f_cost = new_g_cost + new_h_cost
                    new_state.moves = move_states[i]
                    
                    if new_state in open_dict:
                        open_dict[new_state].costs = (
                            new_f_cost, new_g_cost, new_h_cost)
                        open_dict[new_state].parent = explore_state
                        heapq.heapify(open_heap)
                    else:
                        new_state.costs = (new_f_cost, new_g_cost, new_h_cost)
                        new_state.parent = explore_state
                        open_dict[new_state] = new_state
                        heapq.heappush(open_heap, new_state) 
        return
    
    def __greedy(self):
        start_state = self.game
        queue = []
        new_cost = Agent.__calculate_h(start_state)
        self.game.costs = (new_cost, 0, new_cost)
        heapq.heappush(queue, start_state)
        visited = set()

        while len(queue) > 0:
            start_time = time.time()
            explore_state = heapq.heappop(queue)
            end_time = time.time()
            self.time_finding_best_state += end_time - start_time

            self.iter_count += 1
            self.max_storage = max(len(queue) + len(visited), self.max_storage)

            if explore_state.is_goal_state():
                self.__trace_path(explore_state)
                return

            self.max_depth = max(explore_state.costs[1] + 1, self.max_depth)

            start_time = time.time()
            (new_states, move_states) = explore_state.next_states()

            end_time = time.time()
            self.time_generating_new_states += end_time - start_time

            new_g_cost = explore_state.costs[1] + 1
            for i, new_state in enumerate(new_states):
            # for new_state in new_states:
                if new_state not in visited:
                    self.state_count += 1
                    new_state.parent = explore_state

                    start_time = end_time
                    new_cost = Agent.__calculate_h(new_state)
                    new_state.costs = (new_cost, new_g_cost, new_cost)
                    end_time = time.time()
                    self.time_updating_states += end_time-start_time
                    new_state.moves = move_states[i]

                    visited.add(new_state)
                    heapq.heappush(queue, new_state)




    def __bfs(self):
        """
        Simple Breadth First Search algorithm
        """
        start = self.game
        queue = [start]
        visited = set()

        while len(queue) > 0:
            start_time = time.time()
            explore_state = queue.pop(0)
            end_time = time.time()
            self.time_finding_best_state += end_time - start_time

            self.iter_count += 1
            self.max_storage = max(len(queue) + len(visited), self.max_storage)

            if explore_state.is_goal_state():
                self.__trace_path(explore_state)
                return
            
            self.max_depth = max(explore_state.costs[1] + 1, self.max_depth)


            start_time = time.time()
            new_states = explore_state.next_states()
            end_time = time.time()
            self.time_generating_new_states += end_time - start_time

            for new_state in new_states:
                new_g_cost = explore_state.costs[1] + 1
                if new_state not in visited:
                    self.state_count += 1
                    new_state.parent = explore_state
                    new_state.costs = (new_g_cost, new_g_cost, 0)
                    visited.add(new_state)
                    queue.append(new_state)

    def __cost_needs_updating(open, new_state, new_g_cost):
        if new_state not in open:
            return True
        else:
            return new_g_cost < open[new_state].costs[1]

    def print_statistics(self):
        print("# STATISTICS")
        print("# -------------------------------")
        print("# Total Time: ",self.end_time - self.start_time)
        print("# Turns: {:,}".format(len(self.path) - 1))
        print("# Iterations: {:,}".format(self.iter_count))
        print("# States looked at: {:,}".format(self.state_count))
        print("# Max states stored at once: {:,}".format(self.max_storage))
        print("# Max depth: {:,}".format(self.max_depth))
        print("# Time finding best state from queue:",self.time_finding_best_state)
        print("# Time generating new states: ",self.time_generating_new_states)
        print("# Time updating states: ",self.time_updating_states)
