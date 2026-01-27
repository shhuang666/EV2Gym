# Episode Settings Inspector

This tool allows you to inspect the case settings for each episode during training **without actually training any model**. It's useful for understanding what scenarios your RL agent will encounter.

## What It Shows

For each episode, the inspector displays:

- **Number of EVs** that arrive during the episode
- **Initial energy** (battery state of charge at arrival)
- **Target energy** (desired battery state at departure)
- **Arrival and departure times** for each EV
- **Stay duration** at the charging station
- **Battery capacity** of each EV
- **Charging power limits** (max AC/DC charge power, max discharge power)
- **Minimum battery capacity** constraints
- **Charge and discharge efficiency**

## Usage

### Using the Shell Script (Recommended)

```bash
# Inspect 5 episodes with default config
./inspect_episodes.sh

# Inspect 10 episodes with a specific config file
./inspect_episodes.sh --config_file ev2gym/example_config_files/simplePST.yaml --num_episodes 10

# Quiet mode (summary only, no detailed EV information)
./inspect_episodes.sh --quiet --num_episodes 3

# With custom reward and state functions
./inspect_episodes.sh --reward_function my_module:my_reward --state_function my_module:my_state
```

### Using Python Directly

```bash
# Basic usage
uv run inspect_episode_settings.py --num_episodes 5

# With specific config file
uv run inspect_episode_settings.py \
    --config_file ev2gym/example_config_files/V2GProfitPlusLoads.yaml \
    --num_episodes 10

# Quiet mode
uv run inspect_episode_settings.py --num_episodes 3 --quiet

# With custom functions
uv run inspect_episode_settings.py \
    --reward_function my_module:my_reward \
    --state_function my_module:my_state \
    --num_episodes 5
```

## Command-Line Arguments

- `--config_file PATH`: Path to the configuration YAML file (default: `ev2gym/example_config_files/V2GProfitPlusLoads.yaml`)
- `--num_episodes N`: Number of episodes to inspect (default: 5)
- `--verbose`: Print detailed information for each EV (default: enabled)
- `--quiet`: Print only summary statistics, skip detailed EV information
- `--reward_function FUNC`: Custom reward function (format: `module:function`)
- `--state_function FUNC`: Custom state function (format: `module:function`)

## Example Output

```
================================================================================
EPISODE 1/2
================================================================================

Simulation Configuration:
  Simulation Length: 96 steps
  Timescale: 15 minutes per step
  Total Simulation Time: 1440 minutes
  Number of Charging Stations: 2

Episode Summary:
  Total EVs in Episode: 2

Detailed EV Information:
--------------------------------------------------------------------------------
  EV #1:
    EV ID: 0
      Location (CS): 0
      Arrival Time: 15 (step 15)
      Departure Time: 39 (step 39)
      Stay Duration: 24 steps
      Battery Capacity: 50.00 kWh
      Initial Energy: 42.94 kWh (85.9%)
      Target Energy: 50.00 kWh (100.0%)
      Energy Needed: 7.06 kWh
      Max AC Charge Power: 11.00 kW
      Max DC Charge Power: 50.00 kW
      Max Discharge Power: -11.00 kW
      ...

EV Arrival Timeline:
--------------------------------------------------------------------------------
  Step 14: 1 EV(s) arrived
    - EV 0 at CS 0, will depart at step 39
  Step 25: 1 EV(s) arrived
    - EV 0 at CS 1, will depart at step 38

Episode Statistics:
  Average Stay Duration: 18.00 steps (270.00 minutes)
  Average Initial SoC: 86.9%
  Average Target SoC: 100.0%
  Average Energy Needed: 6.53 kWh
```

## How It Works

The inspector:
1. Creates an EV2Gym environment with your specified configuration
2. Runs through each episode with **zero actions** (no charging/discharging)
3. Observes all EVs that arrive during the episode
4. Collects and displays detailed information about each EV
5. Calculates statistics across all episodes

This allows you to see exactly what scenarios your RL agent will face during training, without the overhead of actually training a model.

## Comparison with Training

Unlike `train_stable_baselines.py` which:
- Trains an RL model
- Takes hours to complete
- Focuses on learning optimal policies

The episode inspector:
- **Does not train** any model
- Completes in seconds/minutes
- Focuses on **understanding the problem space**

Use this tool to:
- Verify your configuration is generating reasonable scenarios
- Understand the distribution of EV arrivals
- Check battery capacities and energy requirements
- Debug configuration issues before starting expensive training runs
