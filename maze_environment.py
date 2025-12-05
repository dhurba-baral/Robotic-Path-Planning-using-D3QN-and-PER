import numpy as np
import random

class MazeEnvironment:
    """10x10 maze environment with static and dynamic wall configurations"""
    def __init__(self, size=5, dynamic=False):
        self.size = size
        self.dynamic = dynamic
        self.reset()
    
    def reset(self):
        """Reset the environment"""
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        self.steps = 0
        self.collisions = 0
        
        # Generate walls
        self._generate_initial_walls()
        
        return self._get_state()
    
    def _generate_initial_walls(self):
        """Generate initial wall configuration"""
        self.walls = set()
        # Create initial maze pattern with random walls
        num_walls = random.randint(15, 25)
        for _ in range(num_walls):
            wall = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if wall != tuple(self.agent_pos) and wall != tuple(self.goal_pos):
                self.walls.add(wall)
    
    def _move_one_wall(self):
        """Move exactly one wall to a new random position"""
        if not self.walls:
            return
        
        # remove a random wall
        wall_to_move = random.choice(list(self.walls))
        self.walls.remove(wall_to_move)

        #add a new wall at a different random position
        new_wall = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        if (new_wall != tuple(self.agent_pos) and 
            new_wall != tuple(self.goal_pos) and 
            new_wall not in self.walls):
            self.walls.add(new_wall)
            break
    
    def _get_state(self):
        """Get current state representation"""
        state = np.zeros((self.size, self.size), dtype=np.float32)
        state[self.agent_pos[0], self.agent_pos[1]] = 1.0  # Agent
        state[self.goal_pos[0], self.goal_pos[1]] = 0.5    # Goal
        for wall in self.walls:
            if 0 <= wall[0] < self.size and 0 <= wall[1] < self.size:
                state[wall[0], wall[1]] = -1.0  # Wall
        return state.flatten()
    
    def step(self, action):
        """Execute action: 0=up, 1=right, 2=down, 3=left"""
        self.steps += 1
        
        # In dynamic mode, move exactly one wall every four steps
        if self.dynamic and self.steps % 6 == 0:
            self._move_one_wall()
        
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        move = moves[action]
        
        new_pos = [self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]]
        
        # Check boundaries and walls
        if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size 
            and tuple(new_pos) not in self.walls):
            self.agent_pos = new_pos
            collision = False
        else:
            collision = True
            self.collisions += 1
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = 100.0
            done = True
        elif collision:
            reward = -1.0
            done = False
        else:
            # Distance-based reward
            dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            reward = -0.1 - 0.01 * dist
            done = False
        
        # episode ends after 200 steps
        if self.steps >= 200:
            done = True
        
        return self._get_state(), reward, done, {'collisions': self.collisions}
    
