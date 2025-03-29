import argparse
import math
from typing import List, Tuple

from base import SimulationConfig, Simulation, LocalPlanner, DynamicObstacle, GlobalPlannerType


class PotentialFieldLocalPlanner(LocalPlanner):
    """An implementation of a local planner using potential fields."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.attractive_weight = 1.0
        self.repulsive_weight = 5.0
        self.influence_radius = 5.0  # Cells
        self.force_history = []  # For visualizing forces
        
        # Debug information
        self.debug_info = {}

    def set_global_path(self, path: List[Tuple[int, int]]) -> None:
        """Set the global path from the global planner with additional validation."""
        # Store the original path for debugging
        self.original_path = path
        
        # Ensure the path is not empty
        if not path:
            print("Warning: Empty path received from global planner")
            self.global_path = []
            self.current_path_index = 0
            self.local_target = None
            self.debug_info['path_status'] = 'empty'
            return
            
        # Print path information
        print(f"Received path with {len(path)} waypoints")
        if len(path) >= 2:
            print(f"Path starts at {path[0]} and ends at {path[-1]}")
            
        # Ensure waypoints are valid (no duplicates)
        filtered_path = [path[0]]
        for i in range(1, len(path)):
            if path[i] != path[i-1]:  # Skip duplicate points
                filtered_path.append(path[i])
                
        if len(filtered_path) < len(path):
            print(f"Filtered {len(path) - len(filtered_path)} duplicate waypoints")
        
        # Set the path after validation
        self.global_path = filtered_path
        self.current_path_index = 0
        
        # Make sure we have a valid local target if path exists
        if filtered_path:
            # If path has at least two points, use the second as initial target
            if len(filtered_path) > 1:
                self.local_target = filtered_path[1]
            else:
                self.local_target = filtered_path[0]
            
            self.debug_info['path_status'] = 'valid'
            self.debug_info['path_length'] = len(filtered_path)
        else:
            self.local_target = None
            self.debug_info['path_status'] = 'invalid'

    def get_velocity(
        self, obstacles: List[DynamicObstacle]
    ) -> Tuple[float, float]:  # noqa
        """Calculate velocity using attractive and repulsive forces."""
        # Debug information
        self.debug_info['position'] = self.position
        
        # Check if we have a valid path
        if not self.global_path:
            print("No valid path available")
            self.debug_info['velocity_status'] = 'no path'
            return (0, 0)
            
        # Check if we're at the end of the path
        if self.current_path_index >= len(self.global_path) - 1:
            # If we're close to the final goal, stop
            dx = self.position[0] - self.config.end_point[0]
            dy = self.position[1] - self.config.end_point[1]
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist < 0.5:  # Close enough to goal
                self.debug_info['velocity_status'] = 'reached goal'
                return (0, 0)
                
            # Otherwise, aim for the last waypoint in the path
            target = self.global_path[-1]
            self.debug_info['velocity_status'] = 'toward final waypoint'
            self.debug_info['current_target'] = target
        else:
            # Use the next waypoint as target
            target = self.global_path[self.current_path_index + 1]
            self.debug_info['velocity_status'] = 'following path'
            self.debug_info['current_target'] = target
        
        # Attractive force toward target
        dx_att = target[0] - self.position[0]
        dy_att = target[1] - self.position[1]
        dist_att = math.sqrt(dx_att * dx_att + dy_att * dy_att)
        
        # If we're close to current waypoint, move to next one
        if dist_att < 0.5 and self.current_path_index < len(self.global_path) - 1:
            self.current_path_index += 1
            print(f"Moving to waypoint {self.current_path_index + 1}/{len(self.global_path)}")
            return self.get_velocity(obstacles)

        # Normalize attractive force
        if dist_att > 0:
            dx_att = dx_att / dist_att
            dy_att = dy_att / dist_att
        else:
            # If exactly at target (unlikely but possible), avoid divide by zero
            dx_att = 0
            dy_att = 0

        # Repulsive forces from obstacles
        dx_rep, dy_rep = 0, 0
        for obstacle in obstacles:
            dx = self.position[0] - obstacle.current_pos[0]
            dy = self.position[1] - obstacle.current_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < self.influence_radius and dist > 0:
                # Repulsive force is inversely proportional to distance squared
                force = (
                    self.repulsive_weight
                    * (1.0 / dist - 1.0 / self.influence_radius)
                    * (1.0 / (dist * dist))
                )
                dx_rep += dx / dist * force
                dy_rep += dy / dist * force

        # Combined force
        dx_total = self.attractive_weight * dx_att + dx_rep
        dy_total = self.attractive_weight * dy_att + dy_rep

        # Store force for visualization
        self.force_history.append((self.position, (dx_total, dy_total)))
        if len(self.force_history) > 10:
            self.force_history.pop(0)

        # Normalize and scale by speed
        dist_total = math.sqrt(dx_total * dx_total + dy_total * dy_total)
        if dist_total > 0:
            dx_total = dx_total / dist_total * self.speed
            dy_total = dy_total / dist_total * self.speed

        # Log actual velocity for debugging
        self.debug_info['actual_velocity'] = (dx_total, dy_total)
        
        return (dx_total, dy_total)

    def update(self, obstacles: List[DynamicObstacle]) -> None:
        """Update the agent's position with additional debugging."""
        # Get velocity vector
        velocity = self.get_velocity(obstacles)
        
        # Update position
        new_position = (
            self.position[0] + velocity[0],
            self.position[1] + velocity[1],
        )
        
        # Check if we're actually moving
        dx = new_position[0] - self.position[0]
        dy = new_position[1] - self.position[1]
        movement = math.sqrt(dx * dx + dy * dy)
        
        # Log movement
        self.debug_info['movement'] = movement
        
        # Only update position if we're actually moving
        if movement > 0.001:  # Small threshold to account for floating point errors
            self.position = new_position
            # Update metrics
            self.update_metrics(obstacles)
        else:
            # If we're not moving, check if it's because we're at the goal
            dx_to_goal = self.position[0] - self.config.end_point[0]
            dy_to_goal = self.position[1] - self.config.end_point[1]
            dist_to_goal = math.sqrt(dx_to_goal * dx_to_goal + dy_to_goal * dy_to_goal)
            
            if dist_to_goal < 0.5:  # Close enough to goal
                self.metrics.stop()
                print("Goal reached!")
            else:
                # If we're not at the goal and not moving, we might be stuck
                print(f"Warning: Agent not moving. Current position: {self.position}, Target: {self.global_path[self.current_path_index + 1] if self.current_path_index < len(self.global_path) - 1 else 'End'}")
                
                # If we're stuck and not at the last waypoint, try skipping to the next one
                if self.current_path_index < len(self.global_path) - 2:
                    print("Attempting to recover by skipping to next waypoint")
                    self.current_path_index += 1


def main():
    parser = argparse.ArgumentParser(description="Artificial Potential Field Path Planning")
    
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
        default=100,
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
        default=200,  # Reduced from 500 for better performance with RRT
        help="Number of obstacles"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose debug output"
    )
    
    args = parser.parse_args()
    
    # Map string arguments to enum values
    global_planner_map = {
        "astar": GlobalPlannerType.ASTAR,
        "rrt": GlobalPlannerType.RRT,
        "rrtstar": GlobalPlannerType.RRTSTAR
    }

    # Create configuration
    config = SimulationConfig(
        map_width=args.map_width,
        map_height=args.map_height,
        start_point=(5, 5),
        end_point=(args.map_width - 5, args.map_height - 5),
        num_obstacles=args.num_obstacles,
        obstacle_speed=0.005,
        agent_speed=0.2,
        collision_radius=0.5,
        global_planner_type=global_planner_map[args.global_planner]
    )

    print(f"Starting simulation with {args.global_planner} global planner")
    print(f"Map dimensions: {args.map_width}x{args.map_height}")
    print(f"Number of obstacles: {args.num_obstacles}")
    
    # Create and run simulation
    sim = Simulation(config)

    # Use the potential field planner
    sim.run(PotentialFieldLocalPlanner)


if __name__ == "__main__":
    main()