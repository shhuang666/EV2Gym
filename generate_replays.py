#!/usr/bin/env python3
"""
Generate replay files for evaluation.

This script creates replay files that can be used to evaluate different
algorithms under identical conditions.

Usage:
    python generate_replays.py --config_file <config.yaml> --n_replays 10
"""

import yaml
import os
import argparse
import numpy as np
from ev2gym.models import ev2gym_env


def generate_replays(config_file, n_replays, output_dir=None):
    """
    Generate replay files for evaluation.
    
    Args:
        config_file: Path to configuration YAML file
        n_replays: Number of replay files to generate
        output_dir: Directory to save replay files (optional)
    """
    # Load configuration
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
    
    # Extract scenario name
    scenario = config_file.split("/")[-1].split(".")[0]
    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    
    # Set output directory
    if output_dir is None:
        output_dir = f'./replay/{number_of_charging_stations}cs_{n_transformers}tr_{scenario}/'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {n_replays} replay files...")
    print(f"Configuration: {config_file}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    replay_paths = []
    
    for i in range(n_replays):
        print(f"Generating replay {i+1}/{n_replays}...", end=" ")
        
        # Create environment with random game generation and replay saving enabled
        env = ev2gym_env.EV2Gym(
            config_file=config_file,
            generate_rnd_game=True,  # Generate random scenario
            save_replay=True,         # Save the replay file
            replay_save_path=output_dir,
        )
        
        # Run a simple policy (all actions = 1.0)
        # The specific actions don't matter - we just need to run the simulation
        # to generate the replay file
        for step in range(env.simulation_length):
            actions = np.ones(env.cs)  # Charge at maximum rate
            new_state, reward, done, truncated, _ = env.step(actions, visualize=False)
            
            if done:
                break
        
        # Get the replay path
        replay_path = f"{output_dir}replay_{env.sim_name}.pkl"
        replay_paths.append(replay_path)
        
        print(f"✓ Saved to: {replay_path}")
    
    print("-" * 60)
    print(f"Successfully generated {len(replay_paths)} replay files!")
    print(f"Location: {output_dir}")
    
    return replay_paths


def main():
    parser = argparse.ArgumentParser(description='Generate replay files for EV2Gym evaluation')
    
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--n_replays', type=int, default=10,
                        help='Number of replay files to generate (default: 10)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for replay files (optional)')
    
    args = parser.parse_args()
    
    # Generate replays
    replay_paths = generate_replays(
        config_file=args.config_file,
        n_replays=args.n_replays,
        output_dir=args.output_dir
    )
    
    print("\nReplay files can now be used with evaluator.py")
    print("Example:")
    print(f"  python evaluator.py --config_file {args.config_file} --n_test_cycles {args.n_replays}")


if __name__ == "__main__":
    main()
