import argparse
import math
import random
import numpy as np
from typing import List, Tuple

from base import SimulationConfig, Simulation, LocalPlanner, DynamicObstacle, GlobalPlannerType


class AntColonyOptimizationPlanner(LocalPlanner):
    """An implementation of a local planner using Ant Colony Optimization."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        # ACO parameters
        self.num_ants = 50
        self.pheromone_evaporation = 0.5
        self.pheromone_importance = 1.0
        self.heuristic_importance = 2.0
        self.local_search_radius = 10  # Cells
        self.max_iterations = 5  # Balance between computation time and quality

        # Initialize pheromone grid to cover local area
        grid_size = int(self.local_search_radius * 2)
        self.pheromone_grid = np.ones((grid_size, grid_size)) * 0.1
        self.possible_directions = self._generate_directions(
            8
        )  # 8 possible directions # noqa
        self.best_path = []  # Store current best path
        self.current_iteration = 0

    def _generate_directions(
        self, num_directions: int
    ) -> List[Tuple[float, float]]:  # noqa
        """Generate a list of unit vectors representing possible directions."""
        directions = []
        for i in range(num_directions):
            angle = 2 * math.pi * i / num_directions
            dx = math.cos(angle)
            dy = math.sin(angle)
            directions.append((dx, dy))
        return directions

    def _get_grid_indices(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        center = int(self.local_search_radius)
        grid_x = center + int(x)
        grid_y = center + int(y)

        # Ensure indices are within grid bounds
        grid_x = max(0, min(grid_x, self.pheromone_grid.shape[0] - 1))
        grid_y = max(0, min(grid_y, self.pheromone_grid.shape[1] - 1))

        return grid_x, grid_y

    def _calculate_heuristic(
        self,
        direction: Tuple[float, float],
        target: Tuple[float, float],
        obstacles: List[DynamicObstacle],
    ) -> float:
        """Calculate heuristic value based on goal direction and obstacle avoidance."""  # noqa
        # Position after taking this direction
        new_pos_x = self.position[0] + direction[0] * self.speed
        new_pos_y = self.position[1] + direction[1] * self.speed

        # Distance to target
        dx_target = target[0] - new_pos_x
        dy_target = target[1] - new_pos_y
        dist_to_target = math.sqrt(dx_target**2 + dy_target**2)

        # Direction alignment with target
        target_direction = (
            dx_target / (dist_to_target + 0.001),
            dy_target / (dist_to_target + 0.001),
        )
        alignment = (
            direction[0] * target_direction[0]
            + direction[1] * target_direction[1]  # noqa
        )

        # Obstacle avoidance - higher value means safer
        safety = 1.0
        for obstacle in obstacles:
            dx_obs = new_pos_x - obstacle.current_pos[0]
            dy_obs = new_pos_y - obstacle.current_pos[1]
            dist_to_obs = math.sqrt(dx_obs**2 + dy_obs**2)

            # If very close to obstacle, heavily penalize
            if dist_to_obs < self.config.collision_radius * 2:
                safety *= 0.1
            elif dist_to_obs < self.config.collision_radius * 4:
                safety *= 0.5

        # Combined heuristic - balance between goal-seeking and obstacle avoidance # noqa
        heuristic = (alignment + 1) / 2  # normalize to [0, 1]
        heuristic = (
            heuristic * safety / (dist_to_target + 1)
        )  # closer to target is better

        return heuristic + 0.001  # Small constant to avoid division by zero

    def _ant_search(
        self, target: Tuple[float, float], obstacles: List[DynamicObstacle]
    ) -> List[Tuple[float, float]]:
        """Perform a single ant's search for a path."""
        current_pos = (0, 0)  # Relative to agent's position
        path = [current_pos]

        # Each ant takes several steps
        for _ in range(10):  # Limit path length for computational efficiency
            # Calculate probabilities for each direction
            probs = []
            for direction in self.possible_directions:
                # Calculate next position
                next_pos = (
                    current_pos[0] + direction[0],
                    current_pos[1] + direction[1],
                )
                grid_x, grid_y = self._get_grid_indices(
                    next_pos[0], next_pos[1]
                )  # noqa

                # Get pheromone value
                pheromone = self.pheromone_grid[grid_x, grid_y]

                # Get heuristic value
                heuristic = self._calculate_heuristic(
                    direction, target, obstacles
                )  # noqa

                # Calculate probability according to ACO formula
                probability = (pheromone**self.pheromone_importance) * (
                    heuristic**self.heuristic_importance
                )
                probs.append((direction, probability, next_pos))

            # Normalize probabilities
            total_prob = sum(p[1] for p in probs)
            if total_prob > 0:
                probs = [(d, p / total_prob, np) for d, p, np in probs]

                # Select direction based on probability
                rand_val = random.random()
                cumulative_prob = 0
                selected_direction = self.possible_directions[0]  # Default
                selected_next_pos = (0, 0)

                for direction, prob, next_pos in probs:
                    cumulative_prob += prob
                    if rand_val <= cumulative_prob:
                        selected_direction = direction
                        selected_next_pos = next_pos
                        break

                # Move ant
                current_pos = selected_next_pos
                path.append(current_pos)
            else:
                # If all probabilities are zero (e.g., trapped), choose random direction # noqa
                rand_idx = random.randint(0, len(self.possible_directions) - 1)
                selected_direction = self.possible_directions[rand_idx]
                current_pos = (
                    current_pos[0] + selected_direction[0],
                    current_pos[1] + selected_direction[1],
                )
                path.append(current_pos)

        return path

    def _evaluate_path(
        self,
        path: List[Tuple[float, float]],
        target: Tuple[float, float],
        obstacles: List[DynamicObstacle],
    ) -> float:
        """Evaluate the quality of a path."""
        if not path:
            return 0.0

        # Final position relative to agent
        final_rel_pos = path[-1]
        final_abs_pos = (
            self.position[0] + final_rel_pos[0],
            self.position[1] + final_rel_pos[1],
        )

        # Distance to target
        dx = target[0] - final_abs_pos[0]
        dy = target[1] - final_abs_pos[1]
        dist_to_target = math.sqrt(dx**2 + dy**2)

        # Check for collisions
        for rel_pos in path:
            abs_pos = (
                self.position[0] + rel_pos[0],
                self.position[1] + rel_pos[1],
            )  # noqa
            for obstacle in obstacles:
                dx_obs = abs_pos[0] - obstacle.current_pos[0]
                dy_obs = abs_pos[1] - obstacle.current_pos[1]
                dist_to_obs = math.sqrt(dx_obs**2 + dy_obs**2)

                if dist_to_obs < self.config.collision_radius:
                    return 0.1  # Heavy penalty for collision

        # Quality is inversely proportional to distance
        return 1.0 / (dist_to_target + 1)

    def _update_pheromones(
        self, all_paths: List[Tuple[List[Tuple[float, float]], float]]
    ):
        """Update pheromone levels based on path quality."""
        # Evaporate all pheromones
        self.pheromone_grid *= 1 - self.pheromone_evaporation

        # Deposit new pheromones based on path quality
        for path, quality in all_paths:
            for rel_pos in path:
                grid_x, grid_y = self._get_grid_indices(rel_pos[0], rel_pos[1])
                self.pheromone_grid[grid_x, grid_y] += quality

    def get_velocity(
        self, obstacles: List[DynamicObstacle]
    ) -> Tuple[float, float]:  # noqa
        """Calculate velocity using Ant Colony Optimization."""
        if not self.global_path:
            return (0, 0)

        # Target is either the next waypoint or the final goal
        if self.current_path_index < len(self.global_path) - 1:
            target = self.global_path[self.current_path_index + 1]
        else:
            target = self.config.end_point

        # Check if we've reached the current waypoint
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        dist = math.sqrt(dx**2 + dy**2)

        if dist < 0.1:  # If close to waypoint, move to next one
            if self.current_path_index < len(self.global_path) - 1:
                self.current_path_index += 1
            return self.get_velocity(obstacles)

        # Run ACO algorithm for a few iterations
        if self.current_iteration % self.max_iterations == 0:
            # Collect all ant paths and their quality
            all_paths = []
            best_path = None
            best_quality = 0

            # Let each ant find a path
            for _ in range(self.num_ants):
                path = self._ant_search(target, obstacles)
                quality = self._evaluate_path(path, target, obstacles)
                all_paths.append((path, quality))

                # Keep track of best path
                if quality > best_quality:
                    best_quality = quality
                    best_path = path

            # Update pheromones
            self._update_pheromones(all_paths)

            # Store best path
            if best_path:
                self.best_path = best_path

        self.current_iteration += 1

        # Use the best path found
        if self.best_path and len(self.best_path) > 1:
            # First step of the best path gives our direction
            next_step = self.best_path[1]  # Index 1 because 0 is current pos
            dx = next_step[0]
            dy = next_step[1]

            # Normalize to maintain desired speed
            magnitude = math.sqrt(dx**2 + dy**2)
            if magnitude > 0:
                dx = dx / magnitude * self.speed
                dy = dy / magnitude * self.speed

            return (dx, dy)

        # Fallback: Direct movement toward target if no path found
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        dist = math.sqrt(dx**2 + dy**2)

        if dist > 0:
            dx = dx / dist * self.speed
            dy = dy / dist * self.speed

        return (dx, dy)


def main():
    parser = argparse.ArgumentParser(description="Ant Colony Optimization Path Planning")
    
    parser.add_argument(
        "--global_planner", 
        type=str, 
        choices=["astar", "rrt", "rrtstar"], 
        default="astar",
        help="Global planner algorithm to use"
    )
    
    parser.add_argument(
        "--map_width", 
        type=int, 
        default=80,
        help="Width of the map grid"
    )
    
    parser.add_argument(
        "--map_height", 
        type=int, 
        default=80,
        help="Height of the map grid"
    )
    
    parser.add_argument(
        "--num_obstacles", 
        type=int, 
        default=200,
        help="Number of obstacles"
    )
    
    parser.add_argument(
        "--obstacle_speed", 
        type=float, 
        default=0.005,
        help="Speed of obstacles"
    )
    
    parser.add_argument(
        "--agent_speed", 
        type=float, 
        default=0.2,
        help="Speed of agent"
    )
    
    args = parser.parse_args()
    
    # Map string arguments to enum values
    global_planner_map = {
        "astar": GlobalPlannerType.ASTAR,
        "rrt": GlobalPlannerType.RRT,
        "rrtstar": GlobalPlannerType.RRTSTAR
    }

    config = SimulationConfig(
        map_width=args.map_width,
        map_height=args.map_height,
        start_point=(5, 5),
        end_point=(args.map_width - 5, args.map_height - 5),
        num_obstacles=args.num_obstacles,
        obstacle_speed=args.obstacle_speed,
        agent_speed=args.agent_speed,
        collision_radius=0.5,
        global_planner_type=global_planner_map[args.global_planner]
    )

    print(f"Starting simulation with {args.global_planner} global planner")
    print(f"Map dimensions: {args.map_width}x{args.map_height}")
    print(f"Number of obstacles: {args.num_obstacles}")
    
    sim = Simulation(config)
    sim.run(AntColonyOptimizationPlanner)


if __name__ == "__main__":
    main()