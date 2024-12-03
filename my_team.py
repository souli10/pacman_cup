# myTeam.py
# ---------
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


import random
from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point, PriorityQueue


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveDefensiveAgentNORTH', second='OffensiveDefensiveAgentSOUTH', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    SharedState.reset() # Initialize the shard state

    # The following line is an example only; feel free to change it.
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

# North agent - focuses on upper half of map
class OffensiveDefensiveAgentNORTH(CaptureAgent):
    def register_initial_state(self, game_state):
        # Store starting position and initialize parent class
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def is_in_home_territory(self, game_state):
        # Check if agent is in home territory based on team color
        x = game_state.get_agent_state(self.index).get_position()[0]
        return x <= game_state.get_walls().width // 2 if self.red else x >= game_state.get_walls().width // 2

    def update_shared_state(self, game_state):
        # Update shared game state based on current conditions
        
        # Check if capsule was eaten
        if len(self.get_capsules(game_state)) == 0 and not SharedState.capsule_eaten:
            SharedState.capsule_eaten = True
            SharedState.retreat_counter = 64  # Set retreat timer

        # Update food pellet count
        SharedState.dots_eaten = 20 - len(self.get_food(game_state).as_list())
        
        # Check if scared ghost was eaten
        my_pos = game_state.get_agent_state(self.index).get_position()
        for enemy in [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]:
            if (enemy.scared_timer > 0 and enemy.get_position() and 
                self.get_maze_distance(my_pos, enemy.get_position()) <= 1):
                SharedState.scared_ghost_eaten = True
                SharedState.retreat_counter = 34
        
        # Update retreat counter and defensive mode
        SharedState.retreat_counter = max(SharedState.retreat_counter - 1, 0)
        if SharedState.retreat_counter < 1 and SharedState.capsule_eaten and self.is_in_home_territory(game_state):
            SharedState.in_defensive_mode = True

    def should_retreat(self):
        # Determine if agent should retreat based on game conditions
        return (SharedState.dots_eaten >= 18 or 
                (SharedState.capsule_eaten and SharedState.retreat_counter <= 0) or
                (SharedState.scared_ghost_eaten and SharedState.retreat_counter <= 0))

    def choose_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions: return Directions.STOP

        self.update_shared_state(game_state)
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Switch to defensive mode if needed
        if SharedState.in_defensive_mode:
            return self.defend_action(game_state)
            
        # Try to retreat if conditions met
        if self.should_retreat():
            # Try multiple retreat positions until a valid path is found
            retreat_positions = [
                (12, i) if self.red else (18, i) 
                for i in range(11, 14)  # For NORTH agent
            ] if isinstance(self, OffensiveDefensiveAgentNORTH) else [
                (13, i) if self.red else (19, i) 
                for i in range(4, 7)    # For SOUTH agent
            ]
            
            for target in retreat_positions:
                path = a_star_search(SearchProblem(my_pos, target, game_state, self))
                if path and path[0] != Directions.STOP:
                    return path[0]
            
        # Go for capsule if available
        if not SharedState.capsule_eaten and self.get_capsules(game_state):
            path = a_star_search(SearchProblem(my_pos, self.get_capsules(game_state)[0], game_state, self))
            if path and path[0] != Directions.STOP:
                return path[0]

        # Hunt for food pellets
        food = self.get_food(game_state).as_list()
        if not food: return Directions.STOP

        # Focus on appropriate half based on agent type and team color
        if isinstance(self, OffensiveDefensiveAgentNORTH):
            target_food = [f for f in food if f[1] > game_state.get_walls().height/2]
        else:
            target_food = [f for f in food if f[1] <= game_state.get_walls().height/2 - 1]
        
        # Sort food by distance to try multiple targets
        food_targets = sorted(target_food if target_food else food, 
                            key=lambda x: self.get_maze_distance(my_pos, x))
        
        # Try each food target until a valid path is found
        for target in food_targets[:3]:  # Try top 3 closest food pellets
            path = a_star_search(SearchProblem(my_pos, target, game_state, self))
            if path and path[0] != Directions.STOP:
                return path[0]
        
        # If no path found, move randomly but avoid reversing
        reverse_direction = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        safe_actions = [a for a in legal_actions if a != reverse_direction and a != Directions.STOP]
        return random.choice(safe_actions if safe_actions else legal_actions)
        
    def get_features(self, game_state, action):
        # Calculate features for defensive behavior
        successor = game_state.generate_successor(self.index, action)
        features = {}
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        enemies = [a for a in [successor.get_agent_state(i) for i in self.get_opponents(successor)]
                  if a.is_pacman and a.get_position() is not None]

        features['on_defense'] = 0 if my_state.is_pacman else 1
        features['num_invaders'] = len(enemies)
        features['invader_distance'] = min([self.get_maze_distance(my_pos, a.get_position()) 
                                          for a in enemies]) if enemies else \
                                     self.get_maze_distance(my_pos, (18, random.randint(9,11)) 
                                                          if not self.red else (12, random.randint(11,13)))
        features['stop'] = 1 if action == Directions.STOP else 0
        features['reverse'] = 1 if action == Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction] else 0

        return features

    def get_weights(self):
        # Define weights for defensive features
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 
                'stop': -100, 'reverse': -2}

    def defend_action(self, game_state):
        # Choose best defensive action based on features and weights
        actions = game_state.get_legal_actions(self.index)
        values = [sum(f * self.get_weights()[k] for k, f in self.get_features(game_state, a).items()) 
                 for a in actions]
        return random.choice([a for a, v in zip(actions, values) if v == max(values)])

# South agent - focuses on lower half of map
class OffensiveDefensiveAgentSOUTH(CaptureAgent):
    # Similar to NORTH agent but with different target areas
    # [Previous methods identical to NORTH agent]
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def is_in_home_territory(self, game_state):
        x = game_state.get_agent_state(self.index).get_position()[0]
        return x <= game_state.get_walls().width // 2 if self.red else x >= game_state.get_walls().width // 2

    def update_shared_state(self, game_state):
        if len(self.get_capsules(game_state)) == 0 and not SharedState.capsule_eaten:
            SharedState.capsule_eaten = True
            SharedState.retreat_counter = 64

        SharedState.dots_eaten = 20 - len(self.get_food(game_state).as_list())
        my_pos = game_state.get_agent_state(self.index).get_position()
        for enemy in [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]:
            if (enemy.scared_timer > 0 and enemy.get_position() and 
                self.get_maze_distance(my_pos, enemy.get_position()) <= 1):
                SharedState.scared_ghost_eaten = True
                SharedState.retreat_counter = 34
                
        SharedState.retreat_counter = max(SharedState.retreat_counter - 1, 0)
        if SharedState.retreat_counter < 1 and SharedState.capsule_eaten and self.is_in_home_territory(game_state):
            SharedState.in_defensive_mode = True

    def should_retreat(self):
        return (SharedState.dots_eaten >= 18 or 
                (SharedState.capsule_eaten and SharedState.retreat_counter <= 0) or
                (SharedState.scared_ghost_eaten and SharedState.retreat_counter <= 0))

    def choose_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions: return Directions.STOP

        self.update_shared_state(game_state)
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Switch to defensive mode if needed
        if SharedState.in_defensive_mode:
            return self.defend_action(game_state)
            
        # Try to retreat if conditions met
        if self.should_retreat():
            # Try multiple retreat positions until a valid path is found
            retreat_positions = [
                (12, i) if self.red else (18, i) 
                for i in range(11, 14)  # For NORTH agent
            ] if isinstance(self, OffensiveDefensiveAgentNORTH) else [
                (13, i) if self.red else (19, i) 
                for i in range(4, 7)    # For SOUTH agent
            ]
            
            for target in retreat_positions:
                path = a_star_search(SearchProblem(my_pos, target, game_state, self))
                if path and path[0] != Directions.STOP:
                    return path[0]
            
        # Go for capsule if available
        if not SharedState.capsule_eaten and self.get_capsules(game_state):
            path = a_star_search(SearchProblem(my_pos, self.get_capsules(game_state)[0], game_state, self))
            if path and path[0] != Directions.STOP:
                return path[0]

        # Hunt for food pellets
        food = self.get_food(game_state).as_list()
        if not food: return Directions.STOP

        # Focus on appropriate half based on agent type and team color
        if isinstance(self, OffensiveDefensiveAgentNORTH):
            target_food = [f for f in food if f[1] > game_state.get_walls().height/2]
        else:
            target_food = [f for f in food if f[1] <= game_state.get_walls().height/2 - 1]
        
        # Sort food by distance to try multiple targets
        food_targets = sorted(target_food if target_food else food, 
                            key=lambda x: self.get_maze_distance(my_pos, x))
        
        # Try each food target until a valid path is found
        for target in food_targets[:3]:  # Try top 3 closest food pellets
            path = a_star_search(SearchProblem(my_pos, target, game_state, self))
            if path and path[0] != Directions.STOP:
                return path[0]
        
        # If no path found, move randomly but avoid reversing
        reverse_direction = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        safe_actions = [a for a in legal_actions if a != reverse_direction and a != Directions.STOP]
        return random.choice(safe_actions if safe_actions else legal_actions)

    def get_features(self, game_state, action):
        # Similar to NORTH but with different target positions
        successor = game_state.generate_successor(self.index, action)
        features = {}
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        enemies = [a for a in [successor.get_agent_state(i) for i in self.get_opponents(successor)]
                  if a.is_pacman and a.get_position() is not None]

        features['on_defense'] = 0 if my_state.is_pacman else 1
        features['num_invaders'] = len(enemies)
        features['invader_distance'] = min([self.get_maze_distance(my_pos, a.get_position()) 
                                          for a in enemies]) if enemies else \
                                     self.get_maze_distance(my_pos, (19, random.randint(2,4)) 
                                                          if not self.red else (13, random.randint(4,6)))
        features['stop'] = 1 if action == Directions.STOP else 0
        features['reverse'] = 1 if action == Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction] else 0

        return features

    def get_weights(self):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 
                'stop': -100, 'reverse': -2}

    def defend_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [sum(f * self.get_weights()[k] for k, f in self.get_features(game_state, a).items()) 
                 for a in actions]
        return random.choice([a for a, v in zip(actions, values) if v == max(values)])

    
##########
# Others #
##########

# Shared state class to coordinate behavior between agents
class SharedState:
    retreat_counter = 0      # Countdown for retreat behavior
    capsule_eaten = False    # Tracks if power pellet was consumed
    dots_eaten = 0          # Count of food pellets eaten
    scared_ghost_eaten = False  # Tracks if agent ate a scared ghost
    in_defensive_mode = False   # Indicates if agents should switch to defense

    @classmethod
    def reset(cls):
        # Reset all shared state variables at start of game
        cls.retreat_counter = 0
        cls.capsule_eaten = False
        cls.dots_eaten = 0
        cls.scared_ghost_eaten = False
        cls.in_defensive_mode = False

# A* search problem implementation for pathfinding
class SearchProblem:
    def __init__(self, start_state, goal_state, game_state, agent):
        # Initialize search problem with start/goal states and game info
        self.start = start_state
        self.goal = goal_state
        self.game_state = game_state
        self.walls = game_state.get_walls()
        self.agent = agent
        self.width = game_state.get_walls().width
        self.height = game_state.get_walls().height

    def get_start_state(self):
        return self.start

    def is_goal_state(self, state):
        return state == self.goal

    def get_successors(self, state):
        successors = []
        x, y = state

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.direction_to_vector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            
            if 0 <= next_x < self.width and 0 <= next_y < self.height and not self.walls[next_x][next_y]:
                next_state = (next_x, next_y)
                enemies = [self.game_state.get_agent_state(i) for i in self.agent.get_opponents(self.game_state)]
                
                # Check for nearby non-scared ghosts
                if not any(not a.is_pacman and a.get_position() and a.scared_timer <= 0 and
                        self.agent.get_maze_distance(next_state, a.get_position()) <= 2 
                        for a in enemies):
                    successors.append((next_state, action, 1))

        return successors


# A* search implementation for pathfinding
def a_star_search(problem):
    # Initialize priority queue with start state
    pq = PriorityQueue()
    start = problem.get_start_state()
    # Store state, path, and cost; prioritize by f(n) = g(n) + h(n)
    pq.push((start, [], 0), abs(start[0] - problem.goal[0]) + abs(start[1] - problem.goal[1]))
    visited = {start}  # Track visited states

    while not pq.is_empty():
        state, path, cost = pq.pop()
        # Return path if goal reached
        if problem.is_goal_state(state):
            return path if path else [Directions.STOP]

        # Explore successors
        for next_state, action, step_cost in problem.get_successors(state):
            if next_state not in visited:
                visited.add(next_state)
                new_path = path + [action]
                new_cost = cost + step_cost
                # Calculate Manhattan distance heuristic
                h = abs(next_state[0] - problem.goal[0]) + abs(next_state[1] - problem.goal[1])
                # Push with f(n) = g(n) + h(n) priority
                pq.push((next_state, new_path, new_cost), new_cost + h)
    
    # Return STOP if no path found
    return [Directions.STOP]