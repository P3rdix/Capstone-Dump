import math
import random
from typing import List, Tuple, Set, Optional, Dict

from base import SimulationConfig, GlobalPlanner


class RRTGlobalPlanner(GlobalPlanner):
    """Implementation of Rapidly Exploring Random Trees (RRT) for global path planning."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        # RRT parameters
        self.max_iterations = 5000  # Maximum iterations for RRT
        self.step_size = 10 # Step size for extending the tree
        self.goal_sample_rate = 0.5  # Probability of sampling the goal
        self.goal_bias = True  # Whether to bias sampling toward the goal
        self.debug_info = {}  # Store debug information

    def _distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _nearest_node(self, tree: List[Tuple[int, int]], point: Tuple[int, int]) -> Tuple[int, int]:
        """Find the nearest node in the tree to the given point."""
        return min(tree, key=lambda node: self._distance(node, point))

    def _new_point(self, nearest: Tuple[int, int], random_point: Tuple[int, int]) -> Tuple[int, int]:
        """Generate a new point step_size away from nearest in the direction of random_point."""
        dx = random_point[0] - nearest[0]
        dy = random_point[1] - nearest[1]
        distance = self._distance(nearest, random_point)
        
        if distance <= self.step_size:
            return random_point
        
        # Scale to step_size
        dx = dx / distance * self.step_size
        dy = dy / distance * self.step_size
        
        # Round to nearest integer to ensure grid alignment
        new_x = int(nearest[0] + dx)
        new_y = int(nearest[1] + dy)
        
        # Ensure the point is within bounds
        new_x = max(0, min(self.config.map_width - 1, new_x))
        new_y = max(0, min(self.config.map_height - 1, new_y))
        
        return (new_x, new_y)

    def _is_collision_free(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> bool:
        """Check if the line segment between point1 and point2 is collision-free."""
        # Check if either endpoint is an obstacle
        if point1 in self.obstacles or point2 in self.obstacles:
            return False
        
        # Check if the line passes through any obstacles using Bresenham's algorithm
        x1, y1 = point1
        x2, y2 = point2
        
        # Determine the primary direction and set step sizes
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        # Iterate along the line
        x, y = x1, y1
        while True:
            # Check if current point is an obstacle
            if (x, y) in self.obstacles:
                return False
                
            # Check if we've reached the end
            if x == x2 and y == y2:
                break
                
            # Update position
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True

    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Plan a path using RRT algorithm."""
        print(f"Planning RRT path from {start} to {goal}")
        
        # If start and goal are the same, return simple path
        if start == goal:
            return [start]
            
        # Initialize tree with start node
        tree = [start]
        parent = {start: None}  # Dictionary to track the parent of each node
        
        # Set for quick lookup of nodes in the tree
        tree_set = {start}
        
        iterations = 0
        for iteration in range(self.max_iterations):
            iterations = iteration + 1
            
            # Sample a random point with bias toward the goal
            if random.random() < self.goal_sample_rate:
                random_point = goal
            else:
                if self.goal_bias:
                    # Bias toward goal by sampling in a rectangle around the goal and start
                    min_x = min(start[0], goal[0]) - 5
                    max_x = max(start[0], goal[0]) + 5
                    min_y = min(start[1], goal[1]) - 5
                    max_y = max(start[1], goal[1]) + 5
                    
                    # Ensure bounds
                    min_x = max(0, min_x)
                    max_x = min(self.config.map_width - 1, max_x)
                    min_y = max(0, min_y)
                    max_y = min(self.config.map_height - 1, max_y)
                    
                    random_point = (
                        random.randint(min_x, max_x),
                        random.randint(min_y, max_y)
                    )
                else:
                    # Completely random sampling
                    random_point = (
                        random.randint(0, self.config.map_width - 1),
                        random.randint(0, self.config.map_height - 1)
                    )
            
            # Find nearest node in the tree
            nearest = self._nearest_node(tree, random_point)
            
            # Generate new point in the direction of random point
            new_point = self._new_point(nearest, random_point)
            
            # Skip if this point is already in the tree to avoid duplicates
            if new_point in tree_set:
                continue
                
            # Check if the path to the new point is collision-free
            if self._is_collision_free(nearest, new_point):
                # Add the new point to the tree
                tree.append(new_point)
                tree_set.add(new_point)
                parent[new_point] = nearest
                
                # Check if we can reach the goal from the new point
                if (self._distance(new_point, goal) <= self.step_size and 
                    self._is_collision_free(new_point, goal)):
                    # Add goal to the tree
                    tree.append(goal)
                    tree_set.add(goal)
                    parent[goal] = new_point
                    
                    # Reconstruct the path
                    path = [goal]
                    current = goal
                    while parent[current] is not None:
                        current = parent[current]
                        path.append(current)
                    path.reverse()
                    
                    print(f"RRT path found after {iterations} iterations with {len(path)} waypoints")
                    
                    # Store debug information
                    self.debug_info['iterations'] = iterations
                    self.debug_info['nodes_explored'] = len(tree)
                    self.debug_info['path_length'] = len(path)
                    
                    return self.smooth_path(path)
        
        print(f"No path found after {self.max_iterations} iterations")
        
        # If no path is found but tree expanded, try to find best partial path
        if len(tree) > 1:
            print("Attempting to find partial path")
            # Find the node closest to the goal
            closest_node = min(tree, key=lambda node: self._distance(node, goal))
            
            if closest_node != start:
                # Reconstruct path to this node
                path = [closest_node]
                current = closest_node
                while parent[current] is not None:
                    current = parent[current]
                    path.append(current)
                path.reverse()
                
                print(f"Found partial path with {len(path)} waypoints")
                return path
        
        # If all else fails, return just the start point - not creating a direct path
        return [start]


class RRTStarGlobalPlanner(RRTGlobalPlanner):
    """Implementation of RRT* which improves on RRT by finding more optimal paths."""
    
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.search_radius = 5.0  # Radius for finding neighbors in RRT*
        self.cost = {None: 0.0}  # Cost to reach each node from start
    
    def _get_neighbors(self, tree: List[Tuple[int, int]], tree_set: Set[Tuple[int, int]], 
                       point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find all neighbors of a point within search_radius."""
        return [node for node in tree if node in tree_set and
                self._distance(node, point) <= self.search_radius]
    
    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Plan a path using RRT* algorithm."""
        print(f"Planning RRT* path from {start} to {goal}")
        
        # If start and goal are the same, return simple path
        if start == goal:
            return [start]
            
        # Initialize tree with start node
        tree = [start]
        tree_set = {start}
        parent = {start: None}  # Dictionary to track the parent of each node
        self.cost = {start: 0.0}  # Cost to reach each node from start
        
        iterations = 0
        found_path = False
        
        for iteration in range(self.max_iterations):
            iterations = iteration + 1
            
            # Sample a random point with bias toward the goal
            if random.random() < self.goal_sample_rate:
                random_point = goal
            else:
                if self.goal_bias:
                    # Bias toward goal by sampling in a rectangle around the goal and start
                    min_x = min(start[0], goal[0]) - 5
                    max_x = max(start[0], goal[0]) + 5
                    min_y = min(start[1], goal[1]) - 5
                    max_y = max(start[1], goal[1]) + 5
                    
                    # Ensure bounds
                    min_x = max(0, min_x)
                    max_x = min(self.config.map_width - 1, max_x)
                    min_y = max(0, min_y)
                    max_y = min(self.config.map_height - 1, max_y)
                    
                    random_point = (
                        random.randint(min_x, max_x),
                        random.randint(min_y, max_y)
                    )
                else:
                    # Completely random sampling
                    random_point = (
                        random.randint(0, self.config.map_width - 1),
                        random.randint(0, self.config.map_height - 1)
                    )
            
            # Find nearest node in the tree
            nearest = self._nearest_node(tree, random_point)
            
            # Generate new point in the direction of random point
            new_point = self._new_point(nearest, random_point)
            
            # Skip if this point is already in the tree
            if new_point in tree_set:
                continue
                
            # Check if the path to the new point is collision-free
            if self._is_collision_free(nearest, new_point):
                # Find neighbors within search radius
                neighbors = self._get_neighbors(tree, tree_set, new_point)
                
                # Initialize with nearest as parent
                min_cost = self.cost[nearest] + self._distance(nearest, new_point)
                min_parent = nearest
                
                # Find the best parent based on cost
                for neighbor in neighbors:
                    if self._is_collision_free(neighbor, new_point):
                        cost = self.cost[neighbor] + self._distance(neighbor, new_point)
                        if cost < min_cost:
                            min_cost = cost
                            min_parent = neighbor
                
                # Add the new point to the tree
                tree.append(new_point)
                tree_set.add(new_point)
                parent[new_point] = min_parent
                self.cost[new_point] = min_cost
                
                # Rewire the tree - check if new_point can provide a better path for existing nodes
                for neighbor in neighbors:
                    if neighbor != min_parent:
                        new_cost = self.cost[new_point] + self._distance(new_point, neighbor)
                        if new_cost < self.cost[neighbor] and self._is_collision_free(new_point, neighbor):
                            parent[neighbor] = new_point
                            self.cost[neighbor] = new_cost
                
                # Check if we can reach the goal from the new point
                if (self._distance(new_point, goal) <= self.step_size and 
                    self._is_collision_free(new_point, goal)):
                    
                    # Add goal to the tree if it's better
                    goal_cost = self.cost[new_point] + self._distance(new_point, goal)
                    
                    # Only add goal if it's the first time or if we found a better path
                    if goal not in self.cost or goal_cost < self.cost[goal]:
                        if goal not in tree_set:
                            tree.append(goal)
                            tree_set.add(goal)
                        parent[goal] = new_point
                        self.cost[goal] = goal_cost
                        found_path = True
        
        # Reconstruct the best path if goal was reached
        if goal in parent:
            path = [goal]
            current = goal
            while parent[current] is not None:
                current = parent[current]
                path.append(current)
            path.reverse()
            
            print(f"RRT* path found after {iterations} iterations with {len(path)} waypoints")
            
            # Store debug information
            self.debug_info['iterations'] = iterations
            self.debug_info['nodes_explored'] = len(tree)
            self.debug_info['path_length'] = len(path)
            self.debug_info['path_cost'] = self.cost[goal]
            
            return self.smooth_path(path)
            
        print(f"No path found after {self.max_iterations} iterations")
        
        # If no path is found but tree expanded, try to find best partial path
        if len(tree) > 1:
            print("Attempting to find partial path")
            # Find the node closest to the goal
            closest_node = min(tree, key=lambda node: self._distance(node, goal))
            
            if closest_node != start:
                # Reconstruct path to this node
                path = [closest_node]
                current = closest_node
                while parent[current] is not None:
                    current = parent[current]
                    path.append(current)
                path.reverse()
                
                print(f"Found partial path with {len(path)} waypoints")
                return path
        
        # Just return the start point - removed direct path creation
        return [start]

    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Enhanced path smoothing for RRT*."""
        if len(path) <= 2:
            return path
            
        # Start with a more aggressive smoothing - try to connect non-adjacent nodes
        smoothed_path = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            current = path[current_idx]
            
            # Try to find farthest valid connection
            farthest_valid = current_idx + 1
            for i in range(len(path) - 1, current_idx, -1):
                if self._is_collision_free(current, path[i]):
                    farthest_valid = i
                    break
                    
            # Add the farthest valid point to the smoothed path
            smoothed_path.append(path[farthest_valid])
            current_idx = farthest_valid
            
        # Apply additional smoothing using the parent method if needed
        if len(smoothed_path) > 10:  # Only if path is still complex
            return super().smooth_path(smoothed_path)
            
        return smoothed_path