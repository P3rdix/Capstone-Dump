import pygame

import math
import heapq
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Type, Optional
from enum import Enum

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
GRID_SIZE = 10
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
BLUE = (100, 100, 255)
YELLOW = (255, 255, 100)
PURPLE = (180, 100, 255)
CYAN = (100, 255, 255)
GRAY = (80, 80, 80)
DARK_BLUE = (20, 30, 60)

# Custom colors for artistic rendering
BACKGROUND_COLOR = (15, 20, 35)  # Dark blue-ish background
PATH_COLOR = (140, 220, 255)  # Light blue for paths
OBSTACLE_COLOR = (255, 100, 100)  # Soft red for obstacles
AGENT_COLOR = (100, 230, 210)  # Teal for agent
GRID_COLOR = (40, 45, 60)  # Subtle grid lines
START_COLOR = (100, 255, 130)  # Soft green for start
END_COLOR = (230, 130, 255)  # Soft purple for end


class GlobalPlannerType(Enum):
    """Enum for different global planner types."""
    ASTAR = "A*"
    RRT = "RRT"
    RRTSTAR = "RRT*"

@dataclass
class SimulationConfig:
    """Configuration for the simulation."""

    map_width: int = 40  # Grid width
    map_height: int = 30  # Grid height
    start_point: Tuple[int, int] = (5, 5)
    end_point: Tuple[int, int] = (35, 25)
    num_obstacles: int = 10
    obstacle_speed: float = 0.05  # Grid cells per frame
    agent_speed: float = 0.1  # Grid cells per frame
    collision_radius: float = 0.5  # Grid cells
    global_planner_type: GlobalPlannerType = GlobalPlannerType.ASTAR  # Default to A*



@dataclass
class DynamicObstacle:
    """Representation of a moving obstacle."""

    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    current_pos: Tuple[float, float]
    speed: float
    progress: float = 0.0
    pulse_phase: float = 0.0  # For visual pulsing effect

    def update(self) -> None:
        """Update the obstacle position."""
        if self.progress < 1.0:
            self.progress += self.speed
            if self.progress > 1.0:
                self.progress = 1.0

            # Linear interpolation between start and end positions
            self.current_pos = (
                self.start_pos[0]
                + (self.end_pos[0] - self.start_pos[0]) * self.progress,
                self.start_pos[1]
                + (self.end_pos[1] - self.start_pos[1]) * self.progress,
            )
        else:
            # Reverse direction
            self.start_pos, self.end_pos = self.end_pos, self.start_pos
            self.progress = 0.0

        # Update pulse effect
        self.pulse_phase = (self.pulse_phase + 0.05) % (2 * math.pi)


@dataclass
class BenchmarkMetrics:
    """Metrics for benchmarking the algorithm."""

    start_time: float = 0.0
    end_time: float = 0.0
    num_collisions: int = 0
    path_length: float = 0.0
    steps_taken: int = 0
    replanning_count: int = 0

    def start(self) -> None:
        """Start timing."""
        self.start_time = time.time()

    def stop(self) -> None:
        """Stop timing."""
        self.end_time = time.time()

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time == 0.0:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def report(self) -> str:
        """Generate a report of the metrics."""
        elapsed = self.get_elapsed_time()
        return (
            f"Time: {elapsed:.2f}s\n"
            f"Collisions: {self.num_collisions}\n"
            f"Path Length: {self.path_length:.2f}\n"
            f"Steps: {self.steps_taken}\n"
            f"Replanning Count: {self.replanning_count}"
        )

  
class GlobalPlanner:
    """A* algorithm for global path planning."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.obstacles = set()  # Static obstacles for global planning

    def set_static_obstacles(self, obstacles: Set[Tuple[int, int]]) -> None:
        """Set static obstacles for planning."""
        self.obstacles = obstacles

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells."""
        x, y = node
        neighbors = []

        # 8-directional movement for smoother paths
        for dx, dy in [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]:
            nx, ny = x + dx, y + dy

            # Check bounds
            if (
                0 <= nx < self.config.map_width
                and 0 <= ny < self.config.map_height  # noqa
            ):
                # Check if not an obstacle
                if (nx, ny) not in self.obstacles:
                    # Diagonal movement costs sqrt(2)
                    cost = 1.414 if dx != 0 and dy != 0 else 1.0
                    neighbors.append(((nx, ny), cost))

        return neighbors

    def plan_path(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Plan a path using A* algorithm."""
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        open_set_hash = {start}

        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return self.smooth_path(path)

            for neighbor, cost in self.get_neighbors(current):
                tentative_g_score = g_score[current] + cost

                if (
                    neighbor not in g_score
                    or tentative_g_score < g_score[neighbor]  # noqa
                ):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(
                        neighbor, goal
                    )

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        # No path found
        return []

    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:  # noqa
        """Smooth the path for more natural movement."""
        if len(path) <= 2:
            return path

        # Simple path smoothing by removing redundant waypoints
        result = [path[0]]
        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            next_pt = path[i + 1]

            # If current point is not essential (on same line), skip it
            if (curr[0] - prev[0]) * (next_pt[1] - curr[1]) == (
                next_pt[0] - curr[0]
            ) * (curr[1] - prev[1]):
                continue

            result.append(curr)

        result.append(path[-1])
        return result


class LocalPlanner:
    """Base class for local path planning algorithms."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.position = (
            float(config.start_point[0]),
            float(config.start_point[1]),
        )  # noqa
        self.target = config.end_point
        self.speed = config.agent_speed
        self.global_path = []
        self.current_path_index = 0
        self.local_target = None
        self.metrics = BenchmarkMetrics()
        self.last_position = self.position
        self.trail = []  # Store recent positions for trail effect
        self.pulse_phase = 0.0  # For visual pulsing effect

    def set_global_path(self, path: List[Tuple[int, int]]) -> None:
        """Set the global path from the global planner."""
        self.global_path = path
        self.current_path_index = 0
        if path:
            self.local_target = path[min(1, len(path) - 1)]

    def sense_obstacles(
        self, obstacles: List[DynamicObstacle]
    ) -> List[Tuple[float, float]]:
        """Sense nearby obstacles."""
        # In a real implementation, this might include sensor limitations,
        # field of view constraints, etc.
        return [obstacle.current_pos for obstacle in obstacles]

    def update_metrics(self, obstacles: List[DynamicObstacle]) -> None:
        """Update metrics including collision detection."""
        # Calculate distance moved
        dx = self.position[0] - self.last_position[0]
        dy = self.position[1] - self.last_position[1]
        distance = math.sqrt(dx * dx + dy * dy)
        self.metrics.path_length += distance

        # Check for collisions
        for obstacle in obstacles:
            dx = self.position[0] - obstacle.current_pos[0]
            dy = self.position[1] - obstacle.current_pos[1]
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < self.config.collision_radius:
                self.metrics.num_collisions += 1
                break

        self.last_position = self.position
        self.metrics.steps_taken += 1

        # Update trail
        self.trail.append(self.position)
        if len(self.trail) > 20:  # Keep only recent 20 positions
            self.trail.pop(0)

        # Update pulse effect
        self.pulse_phase = (self.pulse_phase + 0.05) % (2 * math.pi)

    def get_velocity(
        self, obstacles: List[DynamicObstacle]
    ) -> Tuple[float, float]:  # noqa
        """
        Calculate the velocity vector for local planning.
        This is the core method to override in derived classes.
        """
        # Default implementation: Simple move toward next waypoint
        if (
            not self.global_path
            or self.current_path_index >= len(self.global_path) - 1  # noqa
        ):
            return (0, 0)  # No movement if no path or reached end

        # Get next waypoint
        next_waypoint = self.global_path[self.current_path_index + 1]

        # Calculate direction vector
        dx = next_waypoint[0] - self.position[0]
        dy = next_waypoint[1] - self.position[1]

        # Normalize and scale by speed
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 0.1:  # If close to waypoint, move to next one
            self.current_path_index += 1
            return self.get_velocity(obstacles)

        dx = dx / distance * self.speed
        dy = dy / distance * self.speed

        return (dx, dy)

    def update(self, obstacles: List[DynamicObstacle]) -> None:
        """Update the agent's position based on local planning."""
        velocity = self.get_velocity(obstacles)

        # Update position
        self.position = (
            self.position[0] + velocity[0],
            self.position[1] + velocity[1],
        )  # noqa

        # Update metrics
        self.update_metrics(obstacles)

        # Check if reached goal
        dx = self.position[0] - self.config.end_point[0]
        dy = self.position[1] - self.config.end_point[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 0.5:  # Close enough to goal
            self.metrics.stop()


class Simulation:
    """Main simulation class."""

    def __init__(self, config: SimulationConfig = None):
        if config is None:
            config = SimulationConfig()
        self.config = config

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Path Planning Simulation")
        self.clock = pygame.time.Clock()

        # Load custom font if available, fall back to system font
        try:
            self.font = pygame.font.Font("freesansbold.ttf", 16)
            self.title_font = pygame.font.Font("freesansbold.ttf", 24)
        except Exception:
            self.font = pygame.font.SysFont(None, 16)
            self.title_font = pygame.font.SysFont(None, 24)

        # Create global planner based on configuration
        self.global_planner = self._create_global_planner()

        # Placeholder for the local planner - will be set in run()
        self.local_planner = None

        # Create obstacles
        self.obstacles = self.generate_obstacles()

        # Generate initial global path
        self.global_path = self.global_planner.plan_path(
            config.start_point, config.end_point
        )

        # State flags
        self.paused = False
        self.show_debug = True
        self.running = True

        # Visual effects
        self.particles = []  # Particle effects
        self.particle_timer = 0
        self.frame_count = 0
        
    def _create_global_planner(self):
        """Create and return the appropriate global planner based on configuration."""
        if self.config.global_planner_type == GlobalPlannerType.ASTAR:
            from base import GlobalPlanner
            return GlobalPlanner(self.config)
        elif self.config.global_planner_type == GlobalPlannerType.RRT:
            from rrt_planner import RRTGlobalPlanner
            return RRTGlobalPlanner(self.config)
        elif self.config.global_planner_type == GlobalPlannerType.RRTSTAR:
            from rrt_planner import RRTStarGlobalPlanner
            return RRTStarGlobalPlanner(self.config)
        else:
            # Default to A* if invalid type
            from base import GlobalPlanner
            return GlobalPlanner(self.config)

    def generate_obstacles(self) -> List[DynamicObstacle]:
        """Generate random dynamic obstacles."""
        obstacles = []

        for _ in range(self.config.num_obstacles):

            # Generate random start and end positions
            start_x = random.randint(
                self.config.start_point[0], self.config.end_point[0] - 2
            )
            start_y = random.randint(
                self.config.start_point[1], self.config.end_point[1] - 2
            )

            # Generate end position that's a reasonable distance away
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(10, 50)
            end_x = start_x + distance * math.cos(angle)
            end_y = start_y + distance * math.sin(angle)

            # Clamp to map bounds
            end_x = max(1, min(self.config.map_width - 2, end_x))
            end_y = max(1, min(self.config.map_height - 2, end_y))

            obstacle = DynamicObstacle(
                start_pos=(start_x, start_y),
                end_pos=(end_x, end_y),
                current_pos=(start_x, start_y),
                speed=self.config.obstacle_speed,
            )
            obstacles.append(obstacle)

        return obstacles

    def grid_to_pixel(self, grid_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert grid coordinates to pixel coordinates."""
        return (int(grid_pos[0] * GRID_SIZE), int(grid_pos[1] * GRID_SIZE))

    def pixel_to_grid(self, pixel_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert pixel coordinates to grid coordinates."""
        return (pixel_pos[0] / GRID_SIZE, pixel_pos[1] / GRID_SIZE)

    def draw_grid(self) -> None:
        """Draw the grid with a subtle appearance."""
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(
                self.screen, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT), 1
            )  # noqa
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(
                self.screen, GRID_COLOR, (0, y), (SCREEN_WIDTH, y), 1
            )  # noqa

    def draw_path(
        self, path: List[Tuple[int, int]], color: Tuple[int, int, int]
    ) -> None:
        """Draw a path on the screen with glowing effect."""
        if not path:
            return

        # Draw the main path
        points = [self.grid_to_pixel(pos) for pos in path]
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 2)

        # Draw glow effect
        # glow_color = (color[0], color[1], color[2], 50)  # Transparent
        for i in range(len(path) - 1):
            start = self.grid_to_pixel(path[i])
            end = self.grid_to_pixel(path[i + 1])

            # Draw pulsing glow (varies with time)
            pulse = (math.sin(self.frame_count * 0.05) + 1) * 0.5  # 0 to 1
            glow_width = int(4 + pulse * 4)
            pygame.draw.line(
                self.screen,
                (color[0] // 2, color[1] // 2, color[2] // 2, 30),
                start,
                end,
                glow_width,
            )

    def draw_obstacles(self) -> None:
        """Draw the obstacles with a glowing effect."""
        for obstacle in self.obstacles:
            pos = self.grid_to_pixel(obstacle.current_pos)

            # Pulsing effect
            pulse = (math.sin(obstacle.pulse_phase) + 1) * 0.5  # 0 to 1
            radius = int(
                (self.config.collision_radius + pulse * 0.2) * GRID_SIZE
            )  # noqa

            # Inner circle
            pygame.draw.circle(self.screen, OBSTACLE_COLOR, pos, radius)

            # Outer glow
            glow_radius = radius + 5
            glow_surface = pygame.Surface(
                (glow_radius * 2, glow_radius * 2), pygame.SRCALPHA
            )
            pygame.draw.circle(
                glow_surface,
                (255, 100, 100, 30),
                (glow_radius, glow_radius),
                glow_radius,
            )
            self.screen.blit(
                glow_surface, (pos[0] - glow_radius, pos[1] - glow_radius)
            )  # noqa

    def draw_agent(self) -> None:
        """Draw the agent with trail effect."""
        if not self.local_planner:
            return

        # Draw trail
        trail_length = len(self.local_planner.trail)
        for i, pos in enumerate(self.local_planner.trail):
            alpha = int(180 * i / trail_length) if trail_length > 0 else 0
            radius = (
                int(0.6 * GRID_SIZE * i / trail_length)
                if trail_length > 0
                else 1  # noqa
            )
            pixel_pos = self.grid_to_pixel(pos)

            # Create a surface for the semitransparent circle
            surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                surface, (*AGENT_COLOR[:3], alpha), (radius, radius), radius
            )
            self.screen.blit(
                surface, (pixel_pos[0] - radius, pixel_pos[1] - radius)
            )  # noqa

        # Draw agent with pulsing effect
        pos = self.grid_to_pixel(self.local_planner.position)
        pulse = (math.sin(self.local_planner.pulse_phase) + 1) * 0.5  # 0 to 1
        radius = int((0.8 + pulse * 0.2) * GRID_SIZE)

        # Inner circle
        pygame.draw.circle(self.screen, AGENT_COLOR, pos, radius)

        # Outer glow
        glow_radius = radius + 5
        glow_surface = pygame.Surface(
            (glow_radius * 2, glow_radius * 2), pygame.SRCALPHA
        )
        pygame.draw.circle(
            glow_surface,
            (*AGENT_COLOR[:3], 30),
            (glow_radius, glow_radius),
            glow_radius,
        )
        self.screen.blit(
            glow_surface, (pos[0] - glow_radius, pos[1] - glow_radius)
        )  # noqa

        # Draw local target if debugging
        if self.show_debug and self.local_planner.local_target:
            target_pos = self.grid_to_pixel(self.local_planner.local_target)
            pygame.draw.circle(self.screen, YELLOW, target_pos, 5)

    def create_particles(self, position, color, count=5):
        """Create particles at the specified position."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0)
            size = random.uniform(1, 3)
            lifetime = random.uniform(20, 60)
            self.particles.append(
                {
                    "position": position,
                    "velocity": (
                        math.cos(angle) * speed,
                        math.sin(angle) * speed,
                    ),  # noqa
                    "size": size,
                    "color": color,
                    "lifetime": lifetime,
                    "remaining": lifetime,
                }
            )

    def update_particles(self):
        """Update particle positions and lifetimes."""
        for particle in self.particles[:]:
            # Update position
            particle["position"] = (
                particle["position"][0] + particle["velocity"][0],
                particle["position"][1] + particle["velocity"][1],
            )

            # Update lifetime
            particle["remaining"] -= 1
            if particle["remaining"] <= 0:
                self.particles.remove(particle)

    def draw_particles(self):
        """Draw all particles."""
        for particle in self.particles:
            # Calculate alpha based on remaining lifetime
            alpha = int(255 * particle["remaining"] / particle["lifetime"])
            color = (*particle["color"][:3], alpha)

            # Calculate size (particles shrink as they age)
            size = (
                particle["size"] * particle["remaining"] / particle["lifetime"]
            )  # noqa

            # Create surface for transparency
            particle_surface = pygame.Surface(
                (int(size * 2), int(size * 2)), pygame.SRCALPHA
            )
            pygame.draw.circle(
                particle_surface, color, (int(size), int(size)), int(size)
            )

            # Convert grid position to pixel position
            pos = self.grid_to_pixel(particle["position"])
            self.screen.blit(
                particle_surface, (pos[0] - int(size), pos[1] - int(size))
            )  # noqa

    def draw_status(self) -> None:
        """Draw status information with fancy styling."""
        if not self.local_planner:
            return

        # Create translucent panel for status
        panel_width = 200
        panel_height = 180
        panel_surface = pygame.Surface(
            (panel_width, panel_height), pygame.SRCALPHA
        )  # noqa
        panel_surface.fill((20, 20, 40, 180))  # Semi-transparent dark BG

        # Add border
        pygame.draw.rect(
            panel_surface,
            (100, 100, 200, 255),
            (0, 0, panel_width, panel_height),
            1,  # noqa
        )

        # Add title
        title = "STATS"
        title_surface = self.title_font.render(title, True, (180, 180, 255))
        panel_surface.blit(title_surface, (10, 10))

        # Add separator line
        pygame.draw.line(
            panel_surface,
            (100, 100, 200, 255),
            (10, 40),
            (panel_width - 10, 40),
            1,  # noqa
        )

        # Create status text
        metrics = self.local_planner.metrics
        text_lines = [
            f"Time: {metrics.get_elapsed_time():.2f}s",
            f"Collisions: {metrics.num_collisions}",
            f"Path Length: {metrics.path_length:.2f}",
            f"Steps: {metrics.steps_taken}",
            f"Replanning: {metrics.replanning_count}",
        ]

        if self.paused:
            text_lines.append("PAUSED")

        # Draw text with subtle glow effect
        y_offset = 50
        for line in text_lines:
            # Subtle glow
            glow_surface = self.font.render(line, True, (70, 70, 120))
            panel_surface.blit(glow_surface, (12, y_offset + 2))

            # Main text
            text_surface = self.font.render(line, True, (200, 200, 255))
            panel_surface.blit(text_surface, (10, y_offset))
            y_offset += 20

        # Blit the entire panel
        self.screen.blit(panel_surface, (SCREEN_WIDTH - panel_width - 10, 10))

    def draw_title(self) -> None:
        """Draw a title banner for the simulation."""
        # Include global planner type in the title
        title = f"PATH PLANNING SIMULATION - {self.config.global_planner_type.value}"

        # Create gradient background for title
        banner_height = 40
        banner_surface = pygame.Surface(
            (SCREEN_WIDTH, banner_height), pygame.SRCALPHA
        )  # noqa

        # Fill with gradient
        for y in range(banner_height):
            alpha = 180 - int(180 * y / banner_height)
            pygame.draw.line(
                banner_surface, (30, 40, 60, alpha), (0, y), (SCREEN_WIDTH, y)
            )

        # Draw title text with glow effect
        title_shadow = self.title_font.render(title, True, (60, 100, 130))
        title_text = self.title_font.render(title, True, (150, 220, 255))

        # Position text in center
        text_width = title_text.get_width()
        banner_surface.blit(
            title_shadow, ((SCREEN_WIDTH - text_width) // 2 + 2, 12)
        )  # noqa
        banner_surface.blit(title_text, ((SCREEN_WIDTH - text_width) // 2, 10))

        # Add decorative elements
        pygame.draw.line(
            banner_surface,
            (100, 160, 200, 150),
            (50, banner_height - 5),
            (SCREEN_WIDTH - 50, banner_height - 5),
            2,
        )

        # Additional decorative dots
        for x in range(100, SCREEN_WIDTH - 100, 40):
            pygame.draw.circle(
                banner_surface, (100, 160, 200), (x, banner_height - 5), 2
            )

        self.screen.blit(banner_surface, (0, 0))

    def handle_events(self) -> None:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_d:
                    self.show_debug = not self.show_debug
                elif event.key == pygame.K_r:
                    # Reset simulation
                    self.reset()
                elif event.key == pygame.K_g:
                    # Cycle through global planner types
                    self._cycle_global_planner()

    def _cycle_global_planner(self) -> None:
        """Cycle through available global planner types."""
        if self.config.global_planner_type == GlobalPlannerType.ASTAR:
            self.config.global_planner_type = GlobalPlannerType.RRT
        elif self.config.global_planner_type == GlobalPlannerType.RRT:
            self.config.global_planner_type = GlobalPlannerType.RRTSTAR
        elif self.config.global_planner_type == GlobalPlannerType.RRTSTAR:
            self.config.global_planner_type = GlobalPlannerType.ASTAR
            
        # Create new global planner and regenerate path
        self.global_planner = self._create_global_planner()
        self.global_path = self.global_planner.plan_path(
            self.config.start_point, self.config.end_point
        )
        
        # Update local planner with new global path
        if self.local_planner:
            self.local_planner.set_global_path(self.global_path)

    def reset(self) -> None:
        """Reset the simulation."""
        # Reset obstacles
        self.obstacles = self.generate_obstacles()

        # Clear particles
        self.particles = []

        # Reset planner
        if self.local_planner:
            self.local_planner.position = (
                float(self.config.start_point[0]),
                float(self.config.start_point[1]),
            )
            self.local_planner.current_path_index = 0
            self.local_planner.metrics = BenchmarkMetrics()
            self.local_planner.metrics.start()
            self.local_planner.trail = []

        # Regenerate global path
        self.global_path = self.global_planner.plan_path(
            self.config.start_point, self.config.end_point
        )

        if self.local_planner:
            self.local_planner.set_global_path(self.global_path)

    def update(self) -> None:
        """Update simulation state."""
        if self.paused or not self.local_planner:
            return

        # Update obstacles
        for obstacle in self.obstacles:
            obstacle.update()

        # Update agent
        self.local_planner.update(self.obstacles)

        # Occasionally emit particles from agent
        self.particle_timer += 1
        if self.particle_timer >= 5:
            self.particle_timer = 0
            self.create_particles(self.local_planner.position, AGENT_COLOR, 2)

        # Update existing particles
        self.update_particles()

        # Check for collisions and emit particles
        for obstacle in self.obstacles:
            dx = self.local_planner.position[0] - obstacle.current_pos[0]
            dy = self.local_planner.position[1] - obstacle.current_pos[1]
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < self.config.collision_radius + 0.1:
                # Create collision particles
                self.create_particles(
                    self.local_planner.position, (255, 100, 100), 15
                )  # noqa

        # Check if reached goal
        dx = self.local_planner.position[0] - self.config.end_point[0]
        dy = self.local_planner.position[1] - self.config.end_point[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 0.5:  # Close enough to goal
            print("Goal reached!")
            print(self.local_planner.metrics.report())
            self.paused = True

            # Create celebration particles
            for _ in range(3):
                self.create_particles(
                    self.config.end_point, (100, 255, 100), 10
                )  # noqa

        # Update frame counter for animations
        self.frame_count += 1

    def run(self, local_planner_class=LocalPlanner) -> BenchmarkMetrics:
        """Run the simulation with the specified local planner."""
        # Create local planner
        self.local_planner = local_planner_class(self.config)
        self.local_planner.set_global_path(self.global_path)
        self.local_planner.metrics.start()

        # Main loop
        self.running = True
        while self.running:
            self.handle_events()
            self.update()

            # Draw everything
            self.screen.fill(BACKGROUND_COLOR)
            self.draw_grid()

            # Draw path first (so it appears underneath other elements)
            self.draw_path(self.global_path, PATH_COLOR)

            # Draw start and end points
            start_pos = self.grid_to_pixel(self.config.start_point)
            end_pos = self.grid_to_pixel(self.config.end_point)

            # Glowing effect for start/end
            glow_radius = 15 + int(math.sin(self.frame_count * 0.05) * 3)

            # Start point glow
            start_surface = pygame.Surface(
                (glow_radius * 2, glow_radius * 2), pygame.SRCALPHA
            )
            pygame.draw.circle(
                start_surface,
                (*START_COLOR[:3], 50),
                (glow_radius, glow_radius),
                glow_radius,
            )
            self.screen.blit(
                start_surface,
                (start_pos[0] - glow_radius, start_pos[1] - glow_radius),  # noqa
            )

            # End point glow
            end_surface = pygame.Surface(
                (glow_radius * 2, glow_radius * 2), pygame.SRCALPHA
            )
            pygame.draw.circle(
                end_surface,
                (*END_COLOR[:3], 50),
                (glow_radius, glow_radius),
                glow_radius,
            )
            self.screen.blit(
                end_surface,
                (end_pos[0] - glow_radius, end_pos[1] - glow_radius),  # noqa
            )

            # Actual start/end points
            pygame.draw.circle(self.screen, START_COLOR, start_pos, 8)
            pygame.draw.circle(self.screen, END_COLOR, end_pos, 8)

            # Draw inner circles with pulsing effect
            inner_size = 4 + int(math.sin(self.frame_count * 0.1) * 2)
            pygame.draw.circle(
                self.screen, (255, 255, 255), start_pos, inner_size
            )  # noqa
            pygame.draw.circle(
                self.screen, (255, 255, 255), end_pos, inner_size
            )  # noqa

            # Draw obstacles, particles, and agent
            self.draw_obstacles()
            self.draw_particles()
            self.draw_agent()

            # Draw UI elements
            self.draw_title()
            self.draw_status()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        return self.local_planner.metrics
