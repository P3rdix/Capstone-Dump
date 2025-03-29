import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def visualize_benchmark_results(csv_file, output_prefix="benchmark_viz"):
    """Visualize benchmark results from CSV file."""
    # Read the CSV data
    df = pd.read_csv(csv_file)
    
    # Set up plotting style
    plt.style.use('ggplot')
    sns.set_palette("colorblind")
    
    # Create a unique identifier for each planner combination
    df['planner_combo'] = df['global_planner'] + ' + ' + df['local_planner']
    
    # Plot 1: Total time by obstacle count for each planner combo
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df, 
        x='num_obstacles', 
        y='total_time', 
        hue='planner_combo',
        marker='o',
        linewidth=2.5
    )
    plt.title('Total Planning Time by Number of Obstacles', fontsize=16)
    plt.xlabel('Number of Obstacles', fontsize=14)
    plt.ylabel('Total Time (seconds)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Planner Combination', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_total_time.png", dpi=300)
    
    # Plot 2: Stacked bar chart showing global vs local planning time
    plt.figure(figsize=(14, 10))
    
    # Prepare data for stacked bar chart
    pivot_df = df.pivot_table(
        index=['planner_combo', 'num_obstacles'],
        values=['global_planning_time', 'local_planning_time']
    ).reset_index()
    
    # Sort by planner combo and obstacle count
    pivot_df = pivot_df.sort_values(['planner_combo', 'num_obstacles'])
    
    # Create labels for x-axis
    bar_labels = [f"{row['planner_combo']}\n{row['num_obstacles']} obstacles" 
                 for _, row in pivot_df.iterrows()]
    
    # Plot stacked bars
    bar_width = 0.8
    bars1 = plt.bar(
        range(len(pivot_df)), 
        pivot_df['global_planning_time'], 
        bar_width, 
        label='Global Planning Time'
    )
    bars2 = plt.bar(
        range(len(pivot_df)), 
        pivot_df['local_planning_time'], 
        bar_width, 
        bottom=pivot_df['global_planning_time'],
        label='Local Planning Time'
    )
    
    plt.title('Planning Time Breakdown: Global vs Local', fontsize=16)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.xticks(range(len(pivot_df)), bar_labels, rotation=90, fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_time_breakdown.png", dpi=300)
    
    # Plot 3: Success rate by obstacle count
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df, 
        x='num_obstacles', 
        y='reached_goal', 
        hue='planner_combo',
        marker='o',
        linewidth=2.5
    )
    plt.title('Goal Reached Rate by Number of Obstacles', fontsize=16)
    plt.xlabel('Number of Obstacles', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.ylim(0, 105)  # Add a little space above 100%
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Planner Combination', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_success_rate.png", dpi=300)
    
    # Plot 4: Path length comparison
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df, 
        x='num_obstacles', 
        y='path_length', 
        hue='planner_combo',
        marker='o',
        linewidth=2.5
    )
    plt.title('Path Length by Number of Obstacles', fontsize=16)
    plt.xlabel('Number of Obstacles', fontsize=14)
    plt.ylabel('Path Length', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Planner Combination', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_path_length.png", dpi=300)
    
    # Plot 5: Number of collisions
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df, 
        x='num_obstacles', 
        y='num_collisions', 
        hue='planner_combo',
        marker='o',
        linewidth=2.5
    )
    plt.title('Number of Collisions by Number of Obstacles', fontsize=16)
    plt.xlabel('Number of Obstacles', fontsize=14)
    plt.ylabel('Number of Collisions', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Planner Combination', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_collisions.png", dpi=300)
    
    # Plot 6: Replanning Count
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df, 
        x='num_obstacles', 
        y='replanning_count', 
        hue='planner_combo',
        marker='o',
        linewidth=2.5
    )
    plt.title('Replanning Count by Number of Obstacles', fontsize=16)
    plt.xlabel('Number of Obstacles', fontsize=14)
    plt.ylabel('Number of Replans', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Planner Combination', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_replanning.png", dpi=300)
    
    # Plot 7: Heatmap of total time
    plt.figure(figsize=(14, 10))
    
    # Create a pivot table for the heatmap
    heatmap_df = df.pivot_table(
        index='planner_combo',
        columns='num_obstacles',
        values='total_time'
    )
    
    # Plot heatmap
    sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt=".2f", 
        cmap="YlGnBu",
        linewidths=.5,
        cbar_kws={'label': 'Total Time (seconds)'}
    )
    plt.title('Total Planning Time Heatmap', fontsize=16)
    plt.xlabel('Number of Obstacles', fontsize=14)
    plt.ylabel('Planner Combination', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_heatmap.png", dpi=300)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("====================")
    
    # Best combination for each obstacle count
    print("\nFastest Planner Combination by Obstacle Count:")
    for obs_count in df['num_obstacles'].unique():
        subset = df[df['num_obstacles'] == obs_count]
        best_idx = subset['total_time'].idxmin()
        best_combo = subset.loc[best_idx, 'planner_combo']
        best_time = subset.loc[best_idx, 'total_time']
        print(f"  {obs_count} obstacles: {best_combo} ({best_time:.2f}s)")
    
    # Most reliable combination by success rate
    print("\nMost Reliable Planner Combination (Success Rate):")
    for obs_count in df['num_obstacles'].unique():
        subset = df[df['num_obstacles'] == obs_count]
        best_idx = subset['reached_goal'].idxmax()
        best_combo = subset.loc[best_idx, 'planner_combo']
        success_rate = subset.loc[best_idx, 'reached_goal']
        print(f"  {obs_count} obstacles: {best_combo} ({success_rate:.1f}%)")
    
    # Overall average performance
    print("\nOverall Performance by Planner Combination:")
    overall = df.groupby('planner_combo').agg({
        'total_time': 'mean',
        'reached_goal': 'mean',
        'num_collisions': 'mean',
        'replanning_count': 'mean'
    })
    
    for combo, row in overall.iterrows():
        print(f"  {combo}:")
        print(f"    Avg Time: {row['total_time']:.2f}s")
        print(f"    Avg Success: {row['reached_goal']:.1f}%")
        print(f"    Avg Collisions: {row['num_collisions']:.2f}")
        print(f"    Avg Replanning: {row['replanning_count']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    
    parser.add_argument(
        "--input", 
        type=str, 
        default="benchmark_results.csv",
        help="Input CSV file with benchmark results"
    )
    
    parser.add_argument(
        "--output_prefix", 
        type=str, 
        default="benchmark_viz",
        help="Prefix for output visualization files"
    )
    
    args = parser.parse_args()
    
    visualize_benchmark_results(args.input, args.output_prefix)
    print(f"Visualizations saved with prefix '{args.output_prefix}'")