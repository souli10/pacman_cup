# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util
from random import randint
from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point

class SharedState:
    retreat_counter = 0
    capsule_eaten = False
    dots_eaten = 0
    scared_ghost_eaten = False
    in_defensive_mode = False

    @classmethod
    def reset(cls):
        cls.retreat_counter = 0
        cls.capsule_eaten = False
        cls.dots_eaten = 0
        cls.scared_ghost_eaten = False
        cls.in_defensive_mode = False

def create_team(first_index, second_index, is_red,
                first='OffensiveDefensiveAgentNORTH', second='OffensiveDefensiveAgentSOUTH', num_training=0):
    SharedState.reset()
    return [eval(first)(first_index), eval(second)(second_index)]

class OffensiveDefensiveAgentNORTH(CaptureAgent):
    """NORTH agent focused on getting capsule first"""
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.capsule_target = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        capsules = self.get_capsules(game_state)
        if capsules:
            self.capsule_target = capsules[0]

    def is_in_home_territory(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        if self.red:
            return my_pos[0] <= game_state.get_walls().width // 2
        else:
            return my_pos[0] >= game_state.get_walls().width // 2

    def update_shared_state(self, game_state):
        # Check if capsule was just eaten
        capsules = self.get_capsules(game_state)
        if len(capsules) == 0 and not SharedState.capsule_eaten:
            SharedState.capsule_eaten = True
            SharedState.retreat_counter = 60

        # Update dots eaten
        food_left = len(self.get_food(game_state).as_list())
        total_food = 20
        SharedState.dots_eaten = total_food - food_left

        # Check if we ate a scared ghost
        my_pos = game_state.get_agent_state(self.index).get_position()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        
        for enemy in enemies:
            if enemy.scared_timer > 0 and enemy.get_position() is not None:
                if self.get_maze_distance(my_pos, enemy.get_position()) <= 1:
                    SharedState.scared_ghost_eaten = True
                    SharedState.retreat_counter = 20  # Reset counter to 10 moves when we eat a ghost
                    
        if SharedState.retreat_counter > 0:
            SharedState.retreat_counter -= 1
            
        if SharedState.capsule_eaten and self.is_in_home_territory(game_state):
            SharedState.in_defensive_mode = True

    def should_retreat(self, game_state):
        if SharedState.dots_eaten >= 18:
            return True
        if SharedState.capsule_eaten and SharedState.retreat_counter <= 0:
            return True
        if SharedState.scared_ghost_eaten and SharedState.retreat_counter <= 0:
            return True
        return False

    def choose_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return Directions.STOP

        self.update_shared_state(game_state)

        # If we're in defensive mode, check for invaders
        if SharedState.in_defensive_mode:
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            return self.defend_action(game_state)

        # Check retreat conditions
        if self.should_retreat(game_state):
            return self.retreat_action(game_state)

        # Always go for capsule first if not eaten
        capsules = self.get_capsules(game_state)
        if capsules and not SharedState.capsule_eaten:
            return self.capsule_action(game_state)
        
        # After capsule eaten, help with food collection
        return self.food_action(game_state)

    def capsule_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()
        capsules = self.get_capsules(game_state)
        
        if capsules:
            problem = SearchProblem(my_pos, capsules[0], game_state)
            path = a_star_search(problem)
            
            if path and path[0] in legal_actions:
                return path[0]
        return random.choice(legal_actions)

    def food_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()
        food = self.get_food(game_state).as_list()

        if food:
            food_distances = [(self.get_maze_distance(my_pos, food_pos), food_pos) 
                            for food_pos in food]
            _, closest_food = min(food_distances)
            
            problem = SearchProblem(my_pos, closest_food, game_state)
            path = a_star_search(problem)
            
            if path and path[0] in legal_actions:
                return path[0]
        return random.choice(legal_actions)

    def retreat_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()
        
        if self.red:
            defensive_pos = (12, randint(11,13))
        else:
            defensive_pos = (18, randint(9,11))
        
        problem = SearchProblem(my_pos, defensive_pos, game_state)
        path = a_star_search(problem)
        
        if path and path[0] in legal_actions:
            return path[0]
        return random.choice(legal_actions)
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Defensive positioning
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0
        
        # Track invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        
        # If invaders are present, minimize distance to them
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        
        # Defensive positioning when no invaders
        if len(invaders) == 0:
            if not self.red:
                dists = [self.get_maze_distance(my_pos, (18, randint(9,11)))]
                features['invader_distance'] = min(dists)
            else:
                dists = [self.get_maze_distance(my_pos, (12, randint(11,13)))]
                features['invader_distance'] = min(dists)
        
        # Penalize stopping or reversing
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000, 
            'on_defense': 100, 
            'invader_distance': -10, 
            'stop': -100, 
            'reverse': -2
        }

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def defend_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)
    
    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor
    def choose_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return Directions.STOP

        self.update_shared_state(game_state)

        # If we're in defensive mode and there are invaders, use defensive evaluation
        if SharedState.in_defensive_mode:
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            if len(invaders) > 0 or self.is_in_home_territory(game_state):
                return self.defend_action(game_state)

        # Continue with offensive strategy if not defending
        if self.should_retreat(game_state):
            return self.retreat_action(game_state)

        capsules = self.get_capsules(game_state)
        if capsules and not SharedState.capsule_eaten:
            return self.capsule_action(game_state)
        
        return self.food_action(game_state)
    
class OffensiveDefensiveAgentSOUTH(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.is_offensive_mode = True

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def is_in_home_territory(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        if self.red:
            return my_pos[0] <= game_state.get_walls().width // 2
        else:
            return my_pos[0] >= game_state.get_walls().width // 2

    def update_shared_state(self, game_state):
        capsules = self.get_capsules(game_state)
        if len(capsules) == 0 and not SharedState.capsule_eaten:
            SharedState.capsule_eaten = True
            SharedState.retreat_counter = 32

        food_left = len(self.get_food(game_state).as_list())
        total_food = 20
        SharedState.dots_eaten = total_food - food_left

        my_pos = game_state.get_agent_state(self.index).get_position()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        
        for enemy in enemies:
            if enemy.scared_timer > 0 and enemy.get_position() is not None:
                if self.get_maze_distance(my_pos, enemy.get_position()) <= 1:
                    SharedState.scared_ghost_eaten = True
                    SharedState.retreat_counter = 10
                    
        if SharedState.retreat_counter > 0:
            SharedState.retreat_counter -= 1
            
        if SharedState.capsule_eaten and self.is_in_home_territory(game_state):
            SharedState.in_defensive_mode = True

    def should_retreat(self, game_state):
        if SharedState.dots_eaten >= 18:
            return True
        if SharedState.capsule_eaten and SharedState.retreat_counter <= 0:
            return True
        if SharedState.scared_ghost_eaten and SharedState.retreat_counter <= 0:
            return True
        return False

    def choose_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return Directions.STOP

        self.update_shared_state(game_state)

        if SharedState.in_defensive_mode:
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            return self.defend_action(game_state)

        if self.should_retreat(game_state):
            return self.retreat_action(game_state)

        return self.food_action(game_state)

    def food_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()
        food = self.get_food(game_state).as_list()

        if food:
            food_distances = [(self.get_maze_distance(my_pos, food_pos), food_pos) 
                            for food_pos in food]
            _, closest_food = min(food_distances)
            
            problem = SearchProblem(my_pos, closest_food, game_state)
            path = a_star_search(problem)
            
            if path and path[0] in legal_actions:
                return path[0]
        return random.choice(legal_actions)

    def retreat_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()
        
        if self.red:
            defensive_pos = (13, randint(4,6))
        else:
            defensive_pos = (19, randint(2,4))
        
        problem = SearchProblem(my_pos, defensive_pos, game_state)
        path = a_star_search(problem)
        
        if path and path[0] in legal_actions:
            return path[0]
        return random.choice(legal_actions)
    
    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor
        
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        if not SharedState.in_defensive_mode:
            # Offensive features
            features['on_offense'] = 1
            food_list = self.get_food(successor).as_list()
            features['food_count'] = len(food_list)

            if food_list:
                min_dist = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['distance_to_food'] = min_dist

            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
            if ghosts:
                dists = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]
                features['ghost_distance'] = min(dists)

        else:
            # Defensive features
            features['on_defense'] = 1
            if my_state.is_pacman: features['on_defense'] = 0

            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            features['num_invaders'] = len(invaders)

            if len(invaders) > 0:
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                features['invader_distance'] = min(dists)
            
            if len(invaders) == 0:
                if not self.red:
                    dists = [self.get_maze_distance(my_pos, (19, randint(2,4)))]
                    features['invader_distance'] = min(dists)
                else:
                    dists = [self.get_maze_distance(my_pos, (13, randint(4,6)))]
                    features['invader_distance'] = min(dists)

        if action == Directions.STOP: 
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        if not SharedState.in_defensive_mode:
            return {
                'food_count': -100,
                'distance_to_food': -1,
                'ghost_distance': 2,
                'stop': -50,
                'reverse': -2
            }
        else:
            return {
                'num_invaders': -1000,
                'on_defense': 100,
                'invader_distance': -10,
                'stop': -100,
                'reverse': -2
            }

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def defend_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)
    
class SearchProblem:
    def __init__(self, start_state, goal_state, game_state):
        self.start = start_state
        self.goal = goal_state
        self.game_state = game_state
        self.walls = game_state.get_walls()

    def get_start_state(self):
        return self.start

    def is_goal_state(self, state):
        return state == self.goal

    def get_successors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.direction_to_vector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            
            if not self.walls[next_x][next_y]:
                next_state = (next_x, next_y)
                successors.append((next_state, action, 1))
        return successors

def a_star_search(problem, heuristic=None):
    if heuristic is None:
        def heuristic(state, problem):
            goal = problem.goal
            return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

    pq = util.PriorityQueue()
    start_state = problem.get_start_state()
    pq.push((start_state, [], 0), heuristic(start_state, problem))
    visited = set()

    while not pq.is_empty():
        state, path, cost = pq.pop()

        if state not in visited:
            visited.add(state)

            if problem.is_goal_state(state):
                return path

            for next_state, action, step_cost in problem.get_successors(state):
                if next_state not in visited:
                    new_path = path + [action]
                    new_cost = cost + step_cost
                    priority = new_cost + heuristic(next_state, problem)
                    pq.push((next_state, new_path, new_cost), priority)
    return []