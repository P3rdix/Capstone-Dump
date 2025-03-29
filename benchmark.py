import argparse
import csv
import time
from typing import List, Dict, Tuple
import os
import subprocess
import sys
import concurrent.futures
from dataclasses import dataclass

from base import SimulationConfig, GlobalPlannerType, BenchmarkMetrics
from potential_field import PotentialFieldLocalPlanner
from particle_swarm_optimization import ParticleSwarmOptimizationPlanner


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark runs."""
    map_width: int = 100
    map_height: int = 80
    start_point: Tuple[int, int] = (5, 5)
    end_point: Tuple[int, int] = (95, 75)
    agent_speed: float = 0.2
    collision_radius: float = 0.5
    obstacle_speed: float = 0.005
    num_runs: int = 3  # Number of runs to average over
    timeout: int = 300  # Maximum seconds per simulation
    headless: bool = True  # Run without visualization


class HeadlessSimulation:
    """A wrapper for running simulations without visualization for benchmarking."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
    def run(self, local_planner_class):
        """Run the simulation and return metrics."""
        from base import GlobalPlanner
        from rrt_planner import RRTGlobalPlanner, RRTStarGlobalPlanner
        
        # Create appropriate global planner
        if self.config.global_planner_type == GlobalPlannerType.ASTAR:
            global_planner = GlobalPlanner(self.config)
        elif self.config.global_planner_type == GlobalPlannerType.RRT:
            global_planner = RRTGlobalPlanner(self.config)
        elif self.config.global_planner_type == GlobalPlannerType.RRTSTAR:
            global_planner = RRTStarGlobalPlanner(self.config)
        else:
            global_planner = GlobalPlanner(self.config)
        
        # Generate obstacles and set them for the global planner
        from base import DynamicObstacle
        import random
        import math
        
        obstacles = []
        obstacle_positions = set()
        
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
            obstacle_positions.add((int(start_x), int(start_y)))
        
        # Set static obstacles for global planning
        global_planner.set_static_obstacles(obstacle_positions)
        
        # Plan global path
        global_path_start_time = time.time()
        global_path = global_planner.plan_path(
            self.config.start_point, self.config.end_point
        )
        global_path_time = time.time() - global_path_start_time
        
        # Create local planner
        local_planner = local_planner_class(self.config)
        local_planner.set_global_path(global_path)
        
        # Start metrics
        local_planner.metrics.start()
        
        # Simulation loop
        max_iterations = 10000  # Safety limit
        iteration = 0
        
        # We don't need this variable anymore since we're using the built-in metrics
        # Just run the simulation until completion or timeout
        start_time = time.time()  # Just for timeout tracking
        while iteration < max_iterations:
            # Update obstacles
            for obstacle in obstacles:
                obstacle.update()
            
            # Update local planner
            local_planner.update(obstacles)
            
            # Check if we've reached the goal
            dx = local_planner.position[0] - self.config.end_point[0]
            dy = local_planner.position[1] - self.config.end_point[1]
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance < 0.5:  # Close enough to goal
                local_planner.metrics.stop()
                break
                
            # Check for timeout
            if time.time() - start_time > 60:  # 60-second timeout
                local_planner.metrics.stop()
                break
                
            iteration += 1
        
        simulation_metrics = local_planner.metrics
        local_planning_time = simulation_metrics.get_elapsed_time()

        # Return combined metrics
        metrics = {
            "global_planning_time": global_path_time,
            "local_planning_time": local_planning_time,
            "total_time": global_path_time + local_planning_time,
            "path_length": local_planner.metrics.path_length,
            "num_collisions": local_planner.metrics.num_collisions,
            "steps_taken": local_planner.metrics.steps_taken,
            "reached_goal": distance < 0.5
        }
        
        return metrics


def run_benchmark(
    global_planner_type: GlobalPlannerType,
    local_planner_class,
    num_obstacles: int,
    benchmark_config: BenchmarkConfig
) -> Dict:
    """Run a benchmark for a specific configuration and return metrics."""
    
    # Create configuration
    config = SimulationConfig(
        map_width=benchmark_config.map_width,
        map_height=benchmark_config.map_height,
        start_point=benchmark_config.start_point,
        end_point=benchmark_config.end_point,
        num_obstacles=num_obstacles,
        obstacle_speed=benchmark_config.obstacle_speed,
        agent_speed=benchmark_config.agent_speed,
        collision_radius=benchmark_config.collision_radius,
        global_planner_type=global_planner_type
    )
    
    # Run multiple times to get average
    metrics_list = []
    
    for run in range(benchmark_config.num_runs):
        print(f"Running benchmark - Global: {global_planner_type.name}, "
              f"Local: {local_planner_class.__name__}, "
              f"Obstacles: {num_obstacles}, Run: {run+1}/{benchmark_config.num_runs}")
        
        # Create and run simulation
        sim = HeadlessSimulation(config)
        try:
            metrics = sim.run(local_planner_class)
            metrics_list.append(metrics)
        except Exception as e:
            print(f"Error in simulation: {e}")
            # Add failed metrics
            metrics_list.append({
                "global_planning_time": float('nan'),
                "local_planning_time": float('nan'),
                "total_time": float('nan'),
                "path_length": 0,
                "num_collisions": 0,
                "steps_taken": 0,
                "reached_goal": False
            })
    
    # Calculate averages
    if metrics_list:
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if key == "reached_goal":
                # For boolean fields, calculate percentage
                avg_metrics[key] = sum(1 for m in metrics_list if m[key]) / len(metrics_list) * 100
            else:
                # For numeric fields, calculate average
                values = [m[key] for m in metrics_list]
                avg_metrics[key] = sum(values) / len(values)
        
        return avg_metrics
    
    return {
        "global_planning_time": float('nan'),
        "local_planning_time": float('nan'),
        "total_time": float('nan'),
        "path_length": 0,
        "num_collisions": 0,
        "steps_taken": 0,
        "reached_goal": 0
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark path planning algorithms")
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="benchmark_results.csv",
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "--num_runs", 
        type=int, 
        default=3,
        help="Number of runs per configuration to average over"
    )
    
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run benchmarks in parallel"
    )
    
    args = parser.parse_args()
    
    # Benchmark configurations
    global_planners = [
        GlobalPlannerType.ASTAR,
        GlobalPlannerType.RRT,
        GlobalPlannerType.RRTSTAR
    ]
    
    local_planners = [
        PotentialFieldLocalPlanner,
        ParticleSwarmOptimizationPlanner
    ]
    
    obstacle_counts = [50, 100, 250, 500]

    heights = [50,100,250,500,1000]
    widths = [50,100,250,500,1000]
    
    # Create benchmark config
    benchmark_config = BenchmarkConfig(
        num_runs=args.num_runs
    )
    
    # Prepare results
    results = []
    
    # Generate all benchmark configurations
    benchmarks = []
    for global_planner in global_planners:
        for local_planner in local_planners:
            for num_obstacles in obstacle_counts:
                for h in heights:
                    for w in widths:
                        benchmarks.append((global_planner, local_planner, num_obstacles, h, w))
    
    # Run benchmarks
    if args.parallel:
        # Parallel execution
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    run_benchmark, global_planner, local_planner, num_obstacles, benchmark_config
                ): (global_planner, local_planner, num_obstacles)
                for global_planner, local_planner, num_obstacles in benchmarks
            }
            
            for future in concurrent.futures.as_completed(futures):
                global_planner, local_planner, num_obstacles = futures[future]
                try:
                    metrics = future.result()
                    results.append({
                        "global_planner": global_planner.name,
                        "local_planner": local_planner.__name__,
                        "num_obstacles": num_obstacles,
                        **metrics
                    })
                except Exception as e:
                    print(f"Error in benchmark {global_planner.name}/{local_planner.__name__}/{num_obstacles}: {e}")
    else:
        # Sequential execution
        for global_planner, local_planner, num_obstacles, height, width in benchmarks:
            benchmark_config.map_width = width
            benchmark_config.map_height = height
            benchmark_config.end_point = [height - 5, width - 5]
            
            metrics = run_benchmark(global_planner, local_planner, num_obstacles, benchmark_config)
            results.append({
                "global_planner": global_planner.name,
                "local_planner": local_planner.__name__,
                "num_obstacles": num_obstacles,
                "width": width,
                "height": height,
                **metrics
            })
    
    # Write results to CSV
    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = [
            "global_planner", 
            "local_planner", 
            "num_obstacles", 
            "global_planning_time", 
            "local_planning_time", 
            "total_time",
            "path_length",
            "num_collisions",
            "steps_taken",
            "replanning_count",
            "reached_goal"
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in sorted(results, key=lambda x: (
            x["global_planner"], 
            x["local_planner"], 
            x["num_obstacles"],
            x["width"],
            x["height"]
        )):
            writer.writerow(result)
    
    # Also create a tabular view of the total time results
    with open("benchmark_table.csv", 'w', newline='') as csvfile:
        # Set up the table headers: first column is obstacles, 
        # then one column for each global/local planner combo
        fieldnames = ["num_obstacles"]
        for global_planner in global_planners:
            for local_planner in local_planners:
                fieldnames.append(f"{global_planner.name}_{local_planner.__name__}")
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Group results by obstacle count
        obstacle_results = {}
        for num_obstacles in obstacle_counts:
            row = {"num_obstacles": num_obstacles}
            for result in results:
                if result["num_obstacles"] == num_obstacles:
                    key = f"{result['global_planner']}_{result['local_planner']}"
                    row[key] = result["total_time"]
            obstacle_results[num_obstacles] = row
        
        # Write the tabular results
        for num_obstacles in obstacle_counts:
            writer.writerow(obstacle_results[num_obstacles])
    
    # Print completion message
    print(f"Benchmark completed. Results written to {args.output} and benchmark_table.csv")


if __name__ == "__main__":
    main()