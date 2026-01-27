"""
This script inspects and outputs the case settings for each episode during training.
It does not train any model but only shows the EV settings such as:
- Number of EVs
- Initial and target energy
- Arriving and departure time
- Battery capacity
- Charging power limits
etc.
"""

from copy import deepcopy
from ev2gym.rl_agent.reward import (
    SquaredTrackingErrorReward,
    ProfitMax_TrPenalty_UserIncentives,
)
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads
from ev2gym.utilities.loaders import load_function_from_module

import gymnasium as gym
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe


def step_to_time(step, start_hour, start_minute, timescale_minutes):
    """
    Convert a simulation step to 24-hour time format.

    Args:
        step: The simulation step number
        start_hour: Starting hour of simulation (0-23)
        start_minute: Starting minute of simulation (0-59)
        timescale_minutes: Minutes per simulation step

    Returns:
        String in HH:MM format (24-hour)
    """
    total_minutes = start_hour * 60 + start_minute + step * timescale_minutes
    hours = int(total_minutes // 60) % 24
    minutes = int(total_minutes % 60)
    return f"{hours:02d}:{minutes:02d}"


def draw_ev_timeline(all_episode_stats, config):
    """
    Draw a visual timeline of EV arrivals and departures for all episodes.

    Args:
        all_episode_stats: List of episode statistics from inspect_episodes
        config: Configuration dictionary
    """
    num_episodes = len(all_episode_stats)

    # Create figure with subplots for each episode
    fig, axes = plt.subplots(
        num_episodes, 1, figsize=(14, 3 * num_episodes), squeeze=False
    )
    axes = axes.flatten()

    for ep_idx, stats in enumerate(all_episode_stats):
        ax = axes[ep_idx]

        # Get simulation parameters
        sim_length = stats["simulation_length"]
        timescale = stats["timescale"]
        start_hour = config.get("hour", 0)
        start_minute = config.get("minute", 0)

        # Collect EVs by charging station
        cs_evs = {}  # {cs_id: [(arrival, departure, ev_info), ...]}

        for ev_detail in stats["ev_details"]:
            cs_id = ev_detail["location"]
            if cs_id not in cs_evs:
                cs_evs[cs_id] = []
            cs_evs[cs_id].append(ev_detail)

        # Sort charging stations
        cs_ids = sorted(cs_evs.keys())

        if not cs_ids:
            ax.text(
                0.5,
                0.5,
                "No EVs in this episode",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(f"Episode {ep_idx + 1}")
            continue

        # Draw timeline for each charging station
        y_positions = {}
        for i, cs_id in enumerate(cs_ids):
            y_positions[cs_id] = i

        ev_count = 0
        for cs_id in cs_ids:
            y = y_positions[cs_id]

            for ev_detail in cs_evs[cs_id]:
                arrival = ev_detail["arrival_time"]
                departure = ev_detail["departure_time"]
                initial_soc = (
                    ev_detail["initial_energy"] / ev_detail["battery_capacity"] * 100
                )

                # Color based on initial SoC (red=low, green=high)
                soc_normalized = initial_soc / 100
                color = (1 - soc_normalized, soc_normalized, 0.3, 0.8)

                # Draw bar for EV connection period
                bar_height = 0.6
                rect = mpatches.FancyBboxPatch(
                    (arrival, y - bar_height / 2),
                    departure - arrival,
                    bar_height,
                    boxstyle="round,pad=0.02,rounding_size=0.1",
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(rect)

                # Add label with SoC info
                mid_x = (arrival + departure) / 2
                label = f"{initial_soc:.0f}%→100%"
                ax.text(
                    mid_x,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                )

                ev_count += 1

        # Configure axes
        ax.set_xlim(0, sim_length)
        ax.set_ylim(-0.5, len(cs_ids) - 0.5)
        ax.set_yticks(range(len(cs_ids)))
        ax.set_yticklabels([f"CS {cs_id}" for cs_id in cs_ids])

        # Create x-axis labels in 24-hour format
        num_ticks = min(
            13, sim_length // 4 + 1
        )  # Roughly every hour for 15-min timescale
        tick_positions = np.linspace(0, sim_length, num_ticks, dtype=int)
        tick_labels = [
            step_to_time(t, start_hour, start_minute, timescale) for t in tick_positions
        ]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

        ax.set_xlabel("Time (24-hour format)")
        ax.set_ylabel("Charging Station")
        ax.set_title(f"Episode {ep_idx + 1} - {ev_count} EVs")
        ax.grid(True, axis="x", alpha=0.3)

        # Add vertical line for current time reference
        ax.axvline(
            x=0, color="blue", linestyle="--", alpha=0.5, label="Simulation Start"
        )

    # Add legend
    legend_elements = [
        mpatches.Patch(
            facecolor=(1, 0, 0.3, 0.8), edgecolor="black", label="Low Initial SoC (0%)"
        ),
        mpatches.Patch(
            facecolor=(0, 1, 0.3, 0.8),
            edgecolor="black",
            label="High Initial SoC (100%)",
        ),
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 0.99))

    plt.suptitle("EV Arrival and Departure Timeline", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save figure
    output_file = "ev_timeline.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nTimeline visualization saved to: {output_file}")

    plt.show()


def print_ev_info(ev, episode_num, step_num):
    """Print detailed information about an EV."""
    arrival_time_str = step_to_time(
        ev.time_of_arrival, config["hour"], config.get("minute", 0), config["timescale"]
    )
    departure_time_str = step_to_time(
        ev.time_of_departure,
        config["hour"],
        config.get("minute", 0),
        config["timescale"],
    )

    print(f"    EV ID: {ev.id}")
    print(f"      Location (CS): {ev.location}")
    print(f"      Arrival Time: {arrival_time_str} (step {ev.time_of_arrival})")
    print(f"      Departure Time: {departure_time_str} (step {ev.time_of_departure})")
    stay_duration_steps = ev.time_of_departure - ev.time_of_arrival
    print(
        f"      Stay Duration: {stay_duration_steps} steps ({stay_duration_steps * config['timescale'] / 60} hours)"
    )
    print(f"      Battery Capacity: {ev.battery_capacity:.2f} kWh")
    print(
        f"      Initial Energy: {ev.battery_capacity_at_arrival:.2f} kWh ({ev.battery_capacity_at_arrival / ev.battery_capacity * 100:.1f}%)"
    )
    print(
        f"      Target Energy: {ev.desired_capacity:.2f} kWh ({ev.desired_capacity / ev.battery_capacity * 100:.1f}%)"
    )
    print(
        f"      Energy Needed: {ev.desired_capacity - ev.battery_capacity_at_arrival:.2f} kWh"
    )
    print(f"      Max AC Charge Power: {ev.max_ac_charge_power:.2f} kW")
    print(f"      Max DC Charge Power: {ev.max_dc_charge_power:.2f} kW")
    print(f"      Max Discharge Power: {ev.max_discharge_power:.2f} kW")
    print(f"      Min Battery Capacity: {ev.min_battery_capacity:.2f} kWh")
    print(f"      Charge Efficiency: {ev.charge_efficiency}")
    print(f"      Discharge Efficiency: {ev.discharge_efficiency}")
    print()


def inspect_episodes(env, num_episodes, config, verbose=True, show_prices=False):
    """
    Inspect episode settings without training.

    Args:
        env: The EV2Gym environment
        num_episodes: Number of episodes to inspect
        config: Configuration dictionary
        verbose: Whether to print detailed information
        show_prices: Whether to show electricity prices at each step
    """

    # Unwrap the environment to access the underlying EV2Gym environment
    unwrapped_env = env.unwrapped

    all_episode_stats = []

    for episode in range(num_episodes):
        print(f"\n{'=' * 80}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'=' * 80}")

        # Reset environment to get new episode
        obs, info = env.reset()

        episode_stats = {
            "episode_num": episode + 1,
            "simulation_length": config["simulation_length"],
            "timescale": config["timescale"],
            "total_evs": 0,
            "evs_per_step": {},
            "ev_details": [],
            "prices_per_step": {} if show_prices else None,
        }

        if verbose:
            print(f"\nSimulation Configuration:")
            print(f"  Simulation Length: {config['simulation_length']} steps")
            print(f"  Timescale: {config['timescale']} minutes per step")
            print(
                f"  Total Simulation Time: {config['simulation_length'] * config['timescale'] / 60} hours"
            )
            print(f"  Simulation Starting Hour: {config['hour']}:00 (24-hour format)")
            print(
                f"  Number of Charging Stations: {config['number_of_charging_stations']}"
            )
            print()

        # Collect all EVs that will arrive during this episode
        all_evs = []
        step = 0
        done = False

        # Run through the episode to see all EVs
        action_space_size = env.action_space.shape[0]
        zero_actions = np.zeros(action_space_size)
        # one_action = np.ones(action_space_size)

        while not done and step < config["simulation_length"]:
            # Capture prices for this step if requested
            if show_prices and step < config["simulation_length"]:
                # Store prices for each charging station at this step
                step_prices = {}
                for cs in unwrapped_env.charging_stations:
                    step_prices[cs.id] = {
                        "charge_price": unwrapped_env.charge_prices[cs.id, step],
                        "discharge_price": unwrapped_env.discharge_prices[cs.id, step],
                    }
                episode_stats["prices_per_step"][step] = step_prices

            obs, reward, terminated, truncated, info = env.step(deepcopy(zero_actions))
            done = terminated or truncated

            # Check for newly arrived EVs
            for cs in unwrapped_env.charging_stations:
                for ev in cs.evs_connected:
                    if ev is not None:
                        # Check if this EV is new (just arrived)
                        if ev not in all_evs:
                            all_evs.append(ev)

                            # Track EVs per step
                            if step not in episode_stats["evs_per_step"]:
                                episode_stats["evs_per_step"][step] = []
                            episode_stats["evs_per_step"][step].append(ev)

            step += 1

        # Now print summary
        episode_stats["total_evs"] = len(all_evs)

        print(f"\nEpisode Summary:")
        print(f"  Total EVs in Episode: {len(all_evs)}")
        print(f"  Episode Duration: {step} steps")
        print()

        if verbose and len(all_evs) > 0:
            print(f"\nDetailed EV Information:")
            print(f"{'-' * 80}")

            # Sort EVs by arrival time
            all_evs_sorted = sorted(all_evs, key=lambda ev: ev.time_of_arrival)

            for i, ev in enumerate(all_evs_sorted, 1):
                print(f"  EV #{i}:")
                print_ev_info(ev, episode + 1, step)

        # Always collect EV details for timeline visualization
        if len(all_evs) > 0:
            all_evs_sorted = sorted(all_evs, key=lambda ev: ev.time_of_arrival)
            for ev in all_evs_sorted:
                ev_detail = {
                    "ev_id": ev.id,
                    "location": ev.location,
                    "arrival_time": ev.time_of_arrival,
                    "departure_time": ev.time_of_departure,
                    "stay_duration": ev.time_of_departure - ev.time_of_arrival,
                    "battery_capacity": ev.battery_capacity,
                    "initial_energy": ev.battery_capacity_at_arrival,
                    "target_energy": ev.desired_capacity,
                    "energy_needed": ev.desired_capacity
                    - ev.battery_capacity_at_arrival,
                    "max_ac_charge_power": ev.max_ac_charge_power,
                    "max_dc_charge_power": ev.max_dc_charge_power,
                    "max_discharge_power": ev.max_discharge_power,
                    "min_battery_capacity": ev.min_battery_capacity,
                }
                episode_stats["ev_details"].append(ev_detail)

        # Print arrival timeline
        if verbose:
            print(f"\nEV Arrival Timeline:")
            print(f"{'-' * 80}")
            for step_num in sorted(episode_stats["evs_per_step"].keys()):
                evs_at_step = episode_stats["evs_per_step"][step_num]
                print(f"  Step {step_num}: {len(evs_at_step)} EV(s) arrived")
                for ev in evs_at_step:
                    print(
                        f"    - EV {ev.id} at CS {ev.location}, will depart at step {ev.time_of_departure}"
                    )
            print()

        # Print price information if requested
        if show_prices and episode_stats["prices_per_step"]:
            print(f"\nElectricity Prices Timeline (Charge Prices):")
            print(f"{'-' * 80}")
            print(f"  Note: Prices are in EUR/kWh (cost to buy electricity from grid)")
            print()

            # Get unique charging station IDs
            cs_ids = sorted(
                set(
                    cs_id
                    for step_prices in episode_stats["prices_per_step"].values()
                    for cs_id in step_prices.keys()
                )
            )

            # Limit to first 6 charging stations
            display_cs_ids = cs_ids[:6]

            if len(cs_ids) > 6:
                print(f"  Showing first 6 of {len(cs_ids)} charging stations")
                print()

            # Create header with CS columns
            header = f"  {'Step':<6}"
            for cs_id in display_cs_ids:
                header += f" {'CS' + str(cs_id):<15}"
            print(header)

            # Create separator
            separator = f"  {'-' * 6}"
            for _ in display_cs_ids:
                separator += f" {'-' * 15}"
            print(separator)

            # Print prices for each step
            for step_num in sorted(episode_stats["prices_per_step"].keys()):
                row = f"  {step_num:<6}"
                for cs_id in display_cs_ids:
                    prices = episode_stats["prices_per_step"][step_num][cs_id]
                    charge_price = prices["charge_price"]
                    row += f" {charge_price:<15.6f}"
                print(row)

            # Calculate and show statistics
            all_charge_prices = [
                prices["charge_price"]
                for step_prices in episode_stats["prices_per_step"].values()
                for prices in step_prices.values()
            ]
            all_discharge_prices = [
                prices["discharge_price"]
                for step_prices in episode_stats["prices_per_step"].values()
                for prices in step_prices.values()
            ]

            print()
            print(f"  Price Statistics (across all steps and charging stations):")
            print(
                f"    Charge Price - Min: {min(all_charge_prices):.6f}, Max: {max(all_charge_prices):.6f}, Avg: {np.mean(all_charge_prices):.6f} EUR/kWh"
            )
            print(
                f"    Discharge Price - Min: {min(all_discharge_prices):.6f}, Max: {max(all_discharge_prices):.6f}, Avg: {np.mean(all_discharge_prices):.6f} EUR/kWh"
            )
            print()

        # Calculate and print statistics
        if len(all_evs) > 0:
            avg_stay = np.mean(
                [ev.time_of_departure - ev.time_of_arrival for ev in all_evs]
            )
            avg_initial_soc = np.mean(
                [
                    ev.battery_capacity_at_arrival / ev.battery_capacity * 100
                    for ev in all_evs
                ]
            )
            avg_target_soc = np.mean(
                [ev.desired_capacity / ev.battery_capacity * 100 for ev in all_evs]
            )
            avg_energy_needed = np.mean(
                [ev.desired_capacity - ev.battery_capacity_at_arrival for ev in all_evs]
            )

            print(f"Episode Statistics:")
            print(
                f"  Average Stay Duration: {avg_stay:.2f} steps ({avg_stay * config['timescale'] / 60:.2f} hours)"
            )
            print(f"  Average Initial SoC: {avg_initial_soc:.1f}%")
            print(f"  Average Target SoC: {avg_target_soc:.1f}%")
            print(f"  Average Energy Needed: {avg_energy_needed:.2f} kWh")
            print()

            episode_stats["avg_stay_duration"] = avg_stay
            episode_stats["avg_initial_soc"] = avg_initial_soc
            episode_stats["avg_target_soc"] = avg_target_soc
            episode_stats["avg_energy_needed"] = avg_energy_needed

        all_episode_stats.append(episode_stats)

    # Print overall statistics across all episodes
    if num_episodes > 1:
        print(f"\n{'=' * 80}")
        print(f"OVERALL STATISTICS ACROSS {num_episodes} EPISODES")
        print(f"{'=' * 80}")

        total_evs_all = sum([stats["total_evs"] for stats in all_episode_stats])
        avg_evs_per_episode = total_evs_all / num_episodes

        print(f"  Total EVs across all episodes: {total_evs_all}")
        print(f"  Average EVs per episode: {avg_evs_per_episode:.2f}")

        if total_evs_all > 0:
            all_avg_stays = [
                stats["avg_stay_duration"]
                for stats in all_episode_stats
                if "avg_stay_duration" in stats
            ]
            all_avg_initial_socs = [
                stats["avg_initial_soc"]
                for stats in all_episode_stats
                if "avg_initial_soc" in stats
            ]
            all_avg_target_socs = [
                stats["avg_target_soc"]
                for stats in all_episode_stats
                if "avg_target_soc" in stats
            ]
            all_avg_energy_needed = [
                stats["avg_energy_needed"]
                for stats in all_episode_stats
                if "avg_energy_needed" in stats
            ]

            if all_avg_stays:
                print(
                    f"  Overall Average Stay Duration: {np.mean(all_avg_stays):.2f} steps"
                )
                print(
                    f"  Overall Average Initial SoC: {np.mean(all_avg_initial_socs):.1f}%"
                )
                print(
                    f"  Overall Average Target SoC: {np.mean(all_avg_target_socs):.1f}%"
                )
                print(
                    f"  Overall Average Energy Needed: {np.mean(all_avg_energy_needed):.2f} kWh"
                )
        print()

    return all_episode_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect EV2Gym episode settings without training"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="ev2gym/example_config_files/V2GProfitPlusLoads.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--reward_function",
        type=str,
        default=None,
        help="Reward function to use (optional)",
    )
    parser.add_argument(
        "--state_function",
        type=str,
        default=None,
        help="State function to use (optional)",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=5, help="Number of episodes to inspect"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Print only summary information"
    )
    parser.add_argument(
        "--show_prices",
        action="store_true",
        help="Show electricity prices at each step",
    )
    parser.add_argument(
        "--show_timeline",
        action="store_true",
        help="Generate a visual timeline of EV arrivals and departures",
    )

    args = parser.parse_args()
    config_file = args.config_file
    reward_function_arg = args.reward_function
    state_function_arg = args.state_function
    num_episodes = args.num_episodes
    verbose = args.verbose and not args.quiet
    show_prices = args.show_prices
    show_timeline = args.show_timeline

    # Load configuration
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    # Determine reward and state functions based on config file (defaults)
    if config_file == "ev2gym/example_config_files/V2GProfitMax.yaml":
        reward_function = profit_maximization
        state_function = V2G_profit_max
    elif config_file == "ev2gym/example_config_files/PublicPST.yaml":
        reward_function = SquaredTrackingErrorReward
        state_function = PublicPST
    elif config_file == "ev2gym/example_config_files/V2GProfitPlusLoads.yaml":
        reward_function = ProfitMax_TrPenalty_UserIncentives
        state_function = V2G_profit_max_loads
    else:
        # Default for custom config files
        reward_function = profit_maximization
        state_function = V2G_profit_max

    # Override with command-line arguments if provided
    if reward_function_arg is not None:
        reward_function = load_function_from_module(reward_function_arg)

    if state_function_arg is not None:
        state_function = load_function_from_module(state_function_arg)

    print(f"\n{'=' * 80}")
    print("EV2GYM EPISODE SETTINGS INSPECTOR")
    print(f"{'=' * 80}")
    print(f"Configuration File: {config_file}")
    print(f"Reward Function: {reward_function.__name__}")
    print(f"State Function: {state_function.__name__}")
    print(f"Number of Episodes to Inspect: {num_episodes}")
    print(f"{'=' * 80}\n")

    # Register and create environment
    env = gym.make(
        "EV2Gym-v1",
        config_file=config_file,
        reward_function=reward_function,
        state_function=state_function,
        verbose=verbose,
    )

    # Inspect episodes
    stats = inspect_episodes(
        env, num_episodes, config, verbose=verbose, show_prices=show_prices
    )

    # Generate timeline visualization if requested
    if show_timeline:
        draw_ev_timeline(stats, config)

    print(f"\n{'=' * 80}")
    print("INSPECTION COMPLETE")
    print(f"{'=' * 80}\n")
