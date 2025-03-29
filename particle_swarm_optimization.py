import argparse
import math
import random
from typing import List, Tuple

from base import SimulationConfig, Simulation, LocalPlanner, DynamicObstacle, GlobalPlannerType


class ParticleSwarmOptimizationPlanner(LocalPlanner):
    """An implementation of a local planner using Particle Swarm Optimization."""  # noqa

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        # PSO parameters
        self.num_particles = 20
        self.inertia_weight = 0.5
        self.cognitive_weight = 1.5  # Personal best influence
        self.social_weight = 1.5  # Global best influence
        self.max_velocity = self.speed * 1.5
        self.search_radius = 8.0  # Local search radius in cells

        # Initialize particles
        self.particles = []
        self.particle_velocities = []
        self.particle_best_positions = []
        self.particle_best_fitness = []
        self.global_best_position = None
        self.global_best_fitness = float("-inf")

        # Initialize particles in a circle around current position
        self._initialize_particles()

        # For visualization and debugging
        self.best_path = []
        self.iteration_count = 0
        self.update_frequency = 3  # Update PSO every N iterations

    def _initialize_particles(self):
        """Initialize particles in a circle around the current position."""
        self.particles = []
        self.particle_velocities = []
        self.particle_best_positions = []
        self.particle_best_fitness = []

        for i in range(self.num_particles):
            # Random position within search radius
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, self.search_radius)
            dx = distance * math.cos(angle)
            dy = distance * math.sin(angle)

            # Particle position relative to agent
            particle_pos = (dx, dy)
            self.particles.append(particle_pos)

            # Initial velocity is random
            vx = random.uniform(-self.max_velocity, self.max_velocity)
            vy = random.uniform(-self.max_velocity, self.max_velocity)
            self.particle_velocities.append((vx, vy))

            # Initial best position is current position
            self.particle_best_positions.append(particle_pos)
            self.particle_best_fitness.append(float("-inf"))

    def _evaluate_fitness(
        self,
        position: Tuple[float, float],
        target: Tuple[float, float],
        obstacles: List[DynamicObstacle],
    ) -> float:
        """
        Evaluate the fitness of a position based on proximity to target
        and distance from obstacles.
        """
        # Convert relative position to absolute position
        abs_pos = (
            self.position[0] + position[0],
            self.position[1] + position[1],
        )  # noqa

        # Distance to target
        dx_target = target[0] - abs_pos[0]
        dy_target = target[1] - abs_pos[1]
        dist_to_target = math.sqrt(dx_target**2 + dy_target**2)

        # Direction to target (normalized)
        direction_to_target = (
            dx_target / (dist_to_target + 0.001),
            dy_target / (dist_to_target + 0.001),
        )

        # Check for obstacles - calculate minimum distance to any obstacle
        min_obstacle_dist = float("inf")
        for obstacle in obstacles:
            dx_obs = abs_pos[0] - obstacle.current_pos[0]
            dy_obs = abs_pos[1] - obstacle.current_pos[1]
            dist_to_obs = math.sqrt(dx_obs**2 + dy_obs**2)
            min_obstacle_dist = min(min_obstacle_dist, dist_to_obs)

        # Obstacle avoidance reward
        # If too close to obstacle, heavily penalize
        obstacle_reward = 0
        if min_obstacle_dist < self.config.collision_radius:
            obstacle_reward = -1000  # Heavy penalty for collision
        elif min_obstacle_dist < self.config.collision_radius * 3:
            # Penalty decreases as distance increases
            obstacle_reward = -10 / (
                min_obstacle_dist / self.config.collision_radius
            )  # noqa

        # Target proximity reward (negative because we want to minimize distance) # noqa
        target_reward = -dist_to_target

        # Direction alignment reward - higher if particle is pointing toward target # noqa
        particle_direction = (
            position[0]
            / (math.sqrt(position[0] ** 2 + position[1] ** 2) + 0.001),  # noqa
            position[1]
            / (math.sqrt(position[0] ** 2 + position[1] ** 2) + 0.001),  # noqa
        )
        alignment = (
            particle_direction[0] * direction_to_target[0]
            + particle_direction[1] * direction_to_target[1]
        )
        alignment_reward = alignment * 5  # Scale up the importance of alignment # noqa

        # Combined fitness
        fitness = target_reward + obstacle_reward + alignment_reward

        return fitness

    def _update_particles(
        self, target: Tuple[float, float], obstacles: List[DynamicObstacle]
    ):
        """Update particle positions and velocities based on PSO algorithm."""
        for i in range(self.num_particles):
            # Current state
            current_pos = self.particles[i]
            current_vel = self.particle_velocities[i]
            personal_best = self.particle_best_positions[i]

            # Random coefficients
            r1 = random.random()
            r2 = random.random()

            # Update velocity
            # Inertia component
            new_vx = self.inertia_weight * current_vel[0]
            new_vy = self.inertia_weight * current_vel[1]

            # Cognitive component (personal best)
            new_vx += (
                self.cognitive_weight * r1 * (personal_best[0] - current_pos[0])  # noqa
            )  # noqa
            new_vy += (
                self.cognitive_weight * r1 * (personal_best[1] - current_pos[1])  # noqa
            )  # noqa

            # Social component (global best)
            if self.global_best_position:
                new_vx += (
                    self.social_weight
                    * r2
                    * (self.global_best_position[0] - current_pos[0])
                )
                new_vy += (
                    self.social_weight
                    * r2
                    * (self.global_best_position[1] - current_pos[1])
                )

            # Limit velocity
            velocity_magnitude = math.sqrt(new_vx**2 + new_vy**2)
            if velocity_magnitude > self.max_velocity:
                new_vx = new_vx / velocity_magnitude * self.max_velocity
                new_vy = new_vy / velocity_magnitude * self.max_velocity

            self.particle_velocities[i] = (new_vx, new_vy)

            # Update position
            new_x = current_pos[0] + new_vx
            new_y = current_pos[1] + new_vy

            # Limit position to search radius
            pos_magnitude = math.sqrt(new_x**2 + new_y**2)
            if pos_magnitude > self.search_radius:
                new_x = new_x / pos_magnitude * self.search_radius
                new_y = new_y / pos_magnitude * self.search_radius

            self.particles[i] = (new_x, new_y)

            # Evaluate fitness
            fitness = self._evaluate_fitness(
                self.particles[i], target, obstacles
            )  # noqa

            # Update personal best
            if fitness > self.particle_best_fitness[i]:
                self.particle_best_positions[i] = self.particles[i]
                self.particle_best_fitness[i] = fitness

                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_position = self.particles[i]
                    self.global_best_fitness = fitness

    def _build_path_to_best(self):
        """Build a path from current position to global best position."""
        if not self.global_best_position:
            return []

        # Direct path from (0,0) to global best
        path = [(0, 0)]  # Current position (relative)

        # Interpolate a few points along the way
        num_steps = 5
        for i in range(1, num_steps + 1):
            t = i / num_steps
            x = t * self.global_best_position[0]
            y = t * self.global_best_position[1]
            path.append((x, y))

        return path

    def get_velocity(
        self, obstacles: List[DynamicObstacle]
    ) -> Tuple[float, float]:  # noqa
        """Calculate velocity using Particle Swarm Optimization."""
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

        # Run PSO every few iterations to balance computation and reactivity
        if self.iteration_count % self.update_frequency == 0:
            # Reset for changing conditions
            if self.iteration_count % (self.update_frequency * 10) == 0:
                self._initialize_particles()
                self.global_best_fitness = float("-inf")
                self.global_best_position = None

            # Update particles
            self._update_particles(target, obstacles)

            # Build path to best position
            self.best_path = self._build_path_to_best()

        self.iteration_count += 1

        # Use the global best position to determine velocity
        if self.global_best_position:
            # Direction from current position to best position
            dx = self.global_best_position[0]
            dy = self.global_best_position[1]

            # Normalize to maintain desired speed
            magnitude = math.sqrt(dx**2 + dy**2)
            if magnitude > 0:
                dx = dx / magnitude * self.speed
                dy = dy / magnitude * self.speed

            # Avoid obstacles with emergency evasion
            for obstacle in obstacles:
                obs_dx = self.position[0] - obstacle.current_pos[0]
                obs_dy = self.position[1] - obstacle.current_pos[1]
                dist_to_obs = math.sqrt(obs_dx**2 + obs_dy**2)

                # If very close to obstacle, adjust velocity for emergency avoidance # noqa
                if dist_to_obs < self.config.collision_radius * 2:
                    # Direction away from obstacle
                    avoid_dx = obs_dx / dist_to_obs
                    avoid_dy = obs_dy / dist_to_obs

                    # Blend avoidance with original direction
                    avoid_weight = 1.0 - (
                        dist_to_obs / (self.config.collision_radius * 2)
                    )
                    dx = (
                        dx * (1 - avoid_weight)
                        + avoid_dx * avoid_weight * self.speed  # noqa
                    )
                    dy = (
                        dy * (1 - avoid_weight)
                        + avoid_dy * avoid_weight * self.speed  # noqa
                    )

                    # Renormalize
                    magnitude = math.sqrt(dx**2 + dy**2)
                    if magnitude > 0:
                        dx = dx / magnitude * self.speed
                        dy = dy / magnitude * self.speed

            return (dx, dy)

        # Fallback: Direct movement toward target if no best position found
        if dist > 0:
            dx = dx / dist * self.speed
            dy = dy / dist * self.speed

        return (dx, dy)


def main():
    parser = argparse.ArgumentParser(description="Particle Swarm Optimization Path Planning")
    
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

    # Create configuration
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
    
    # Create and run simulation
    sim = Simulation(config)

    # Use the PSO planner
    sim.run(ParticleSwarmOptimizationPlanner)


if __name__ == "__main__":
    main()