# Evaluator Documentation

Complete guide for `evaluator.py` - the EV2Gym evaluation framework for benchmarking EV charging control algorithms.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Usage Guide](#usage-guide)
4. [Custom Functions](#custom-functions)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Output Files](#output-files)
7. [Code Structure](#code-structure)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose

`evaluator.py` is a comprehensive evaluation framework that:
- **Benchmarks** different EV charging control algorithms
- **Ensures fair comparison** using replay files (identical scenarios)
- **Generates comprehensive statistics** and performance metrics
- **Supports multiple algorithm types**: heuristics, MPC, RL, optimization

### Key Features

✅ **Fair Comparison** - Uses replay files to ensure all algorithms face identical scenarios  
✅ **Flexible Functions** - Supports both built-in and custom reward/state functions  
✅ **Multiple Algorithms** - Compare heuristics, RL agents, MPC, and optimization methods  
✅ **Comprehensive Metrics** - Tracks 15+ performance indicators  
✅ **Automated Reporting** - Generates CSV data, LaTeX tables, and statistics  

### Architecture

```
evaluator.py
├── Configuration Loading
├── Function Setup (reward/state)
├── Replay Management
│   ├── Load existing replays
│   └── Generate new replays
├── Algorithm Evaluation Loop
│   ├── RL Algorithms (PPO, SAC, etc.)
│   ├── Heuristic Methods
│   ├── MPC Controllers
│   └── Optimization Solvers
├── Results Collection
└── Export & Visualization
```

---

## Quick Start

### Basic Usage (with defaults)

```bash
python evaluator.py \
    --config_file ev2gym/example_config_files/simplePST.yaml \
    --n_test_cycles 10
```

**Defaults:**
- Reward function: `profit_maximization`
- State function: `V2G_profit_max`

### With Specific Functions

```bash
python evaluator.py \
    --config_file ev2gym/example_config_files/PublicPST.yaml \
    --reward_function SquaredTrackingErrorReward \
    --state_function PublicPST \
    --n_test_cycles 20
```

---

## Usage Guide

### Command-Line Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--config_file` | Path to YAML configuration file | - | Yes |
| `--n_test_cycles` | Number of evaluation episodes | 50 | No |
| `--reward_function` | Reward function (built-in or custom) | `profit_maximization` | No |
| `--state_function` | State function (built-in or custom) | `V2G_profit_max` | No |

### Built-in Functions

#### Reward Functions
- **`profit_maximization`** (default) - Maximize charging station profits
- **`SquaredTrackingErrorReward`** - Minimize power tracking error
- **`ProfitMax_TrPenalty_UserIncentives`** - Profit max with transformer penalties

#### State Functions
- **`V2G_profit_max`** (default) - V2G profit maximization state
- **`PublicPST`** - Public power setpoint tracking state
- **`V2G_profit_max_loads`** - V2G with inflexible loads state

### Common Scenarios

#### Scenario 1: Power Setpoint Tracking
```bash
python evaluator.py \
    --config_file ev2gym/example_config_files/PublicPST.yaml \
    --reward_function SquaredTrackingErrorReward \
    --state_function PublicPST \
    --n_test_cycles 10
```

#### Scenario 2: V2G Profit Maximization
```bash
python evaluator.py \
    --config_file ev2gym/example_config_files/V2GProfitMax.yaml \
    --reward_function profit_maximization \
    --state_function V2G_profit_max \
    --n_test_cycles 10
```

#### Scenario 3: V2G with Loads
```bash
python evaluator.py \
    --config_file ev2gym/example_config_files/V2GProfitPlusLoads.yaml \
    --reward_function ProfitMax_TrPenalty_UserIncentives \
    --state_function V2G_profit_max_loads \
    --n_test_cycles 10
```

### Matching Training Configuration

Check your saved model directory to identify the functions used during training:

```
saved_models/2cs_simplePST/ppo_profit_maximization_V2G_profit_max/
                           ^^^  ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^
                           |    |                    |
                           |    reward function      state function
                           algorithm
```

Then use those same functions for evaluation:
```bash
python evaluator.py \
    --config_file ev2gym/example_config_files/simplePST.yaml \
    --reward_function profit_maximization \
    --state_function V2G_profit_max \
    --n_test_cycles 10
```

---

## Custom Functions

### Overview

The evaluator supports custom reward and state functions using the `module:function` syntax, allowing you to evaluate models with any custom logic without modifying the evaluator code.

### Syntax

- **Built-in function**: Just the function name
  - Example: `profit_maximization`
  
- **Custom function**: `module_path:function_name`
  - Example: `my_rewards:custom_reward`
  - Example: `ev2gym.custom.functions:my_reward`

### Using Custom Functions

#### Example 1: Custom Reward in Local Module

If you have a file `my_rewards.py` in your working directory:

```bash
python evaluator.py \
    --config_file my_config.yaml \
    --reward_function my_rewards:custom_reward \
    --n_test_cycles 10
```

#### Example 2: Custom Functions in Package

If you have `ev2gym/custom/my_functions.py`:

```bash
python evaluator.py \
    --config_file my_config.yaml \
    --reward_function ev2gym.custom.my_functions:my_reward \
    --state_function ev2gym.custom.my_functions:my_state \
    --n_test_cycles 10
```

#### Example 3: Mix Built-in and Custom

```bash
python evaluator.py \
    --config_file my_config.yaml \
    --reward_function profit_maximization \
    --state_function my_states:advanced_state \
    --n_test_cycles 10
```

### Creating Custom Functions

#### Custom Reward Function

Create `my_rewards.py`:

```python
def custom_reward(env, action, next_state, done, info):
    """
    Custom reward function.
    
    Args:
        env: The EV2Gym environment
        action: Action taken (numpy array)
        next_state: Next state (numpy array)
        done: Whether episode is done (bool)
        info: Additional info dictionary
    
    Returns:
        reward: Float reward value
    """
    # Example: Combine profit and user satisfaction
    profit = info.get('total_profits', 0)
    satisfaction = info.get('average_user_satisfaction', 0)
    
    reward = profit + satisfaction * 100
    return reward
```

#### Custom State Function

Create `my_states.py`:

```python
import numpy as np

def custom_state(env):
    """
    Custom state function.
    
    Args:
        env: The EV2Gym environment
    
    Returns:
        state: Numpy array representing the state
    """
    # Example: Combine multiple state features
    soc = env.get_soc()  # State of charge for all EVs
    prices = env.get_current_prices()  # Current electricity prices
    time_features = env.get_time_features()  # Time of day, etc.
    
    state = np.concatenate([soc, prices, time_features])
    return state
```

#### Using Your Custom Functions

```bash
python evaluator.py \
    --config_file my_config.yaml \
    --reward_function my_rewards:custom_reward \
    --state_function my_states:custom_state \
    --n_test_cycles 10
```

---

## Evaluation Metrics

The evaluator tracks comprehensive performance metrics for each algorithm:

### Economic Metrics
- **`total_profits`** - Total revenue minus costs ($)
- **`total_energy_charged`** - Total energy delivered to EVs (kWh)
- **`total_energy_discharged`** - Total energy from V2G (kWh)

### User Metrics
- **`total_ev_served`** - Number of EVs successfully served
- **`average_user_satisfaction`** - Mean user satisfaction score (0-1)
- **`energy_user_satisfaction`** - Energy-weighted satisfaction (kWh)

### Grid Metrics
- **`power_tracker_violation`** - Power setpoint violations (kW)
- **`tracking_error`** - Mean absolute tracking error (kW)
- **`energy_tracking_error`** - Cumulative energy error (kWh)
- **`total_transformer_overload`** - Transformer capacity violations (kW)

### Battery Health
- **`battery_degradation`** - Total battery degradation (%)
- **`battery_degradation_calendar`** - Calendar aging component (%)
- **`battery_degradation_cycling`** - Cycling aging component (%)

### Performance
- **`total_reward`** - Cumulative reward over episode
- **`time`** - Execution time (seconds)

### Statistical Analysis

Results are grouped by algorithm with:
- **Mean** - Average performance across test cycles
- **Std** - Standard deviation (variability)

---

## Output Files

### Directory Structure

```
results/
└── eval_{cs}cs_{tr}tr_{scenario}_{n_algos}_algos_{n_cycles}_exp_{timestamp}/
    ├── {scenario}.yaml              # Configuration copy
    ├── data.csv                      # Raw results
    ├── results_grouped.txt           # LaTeX-formatted summary
    └── plot_results_dict.pkl         # Environment states for plotting
```

### File Descriptions

#### `data.csv`
Raw evaluation data with columns:
- `run` - Test cycle index
- `Algorithm` - Algorithm name
- `control_horizon` - MPC horizon (if applicable)
- All performance metrics (see [Evaluation Metrics](#evaluation-metrics))

**Example:**
```csv
run,Algorithm,total_profits,total_ev_served,average_user_satisfaction,...
0,PPO,1234.56,45,0.92,...
1,PPO,1256.78,47,0.94,...
0,SAC,1189.34,43,0.89,...
```

#### `results_grouped.txt`
LaTeX-formatted table with mean ± std for each algorithm and metric.

**Example:**
```latex
\begin{tabular}{lrr}
\toprule
{} & \multicolumn{2}{r}{total_profits} \\
{} &        mean &         std \\
Algorithm &             &             \\
\midrule
PPO &  1245.67 &   23.45 \\
SAC &  1198.23 &   31.67 \\
\bottomrule
\end{tabular}
```

#### `plot_results_dict.pkl`
Pickled dictionary containing final environment states for each algorithm, used for generating comparison plots.

---

## Code Structure

### Main Function: `evaluator()`

```python
def evaluator():
    # 1. Setup (Lines 41-72)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = arg_parser()
    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
    
    # 2. Function Loading (Lines 73-121)
    # Load reward and state functions (built-in or custom)
    
    # 3. Replay Management (Lines 122-180)
    # Load or generate replay files
    
    # 4. Algorithm Loop (Lines 182-320)
    for algorithm in algorithms:
        for k in range(n_test_cycles):
            # Run simulation
            # Collect metrics
    
    # 5. Results Export (Lines 322-340)
    # Save data, compute statistics
```

### Supported Algorithms

Currently enabled in the evaluator (lines 155-161):
```python
algorithms = [
    ChargeAsFastAsPossible,
    ChargeAsLateAsPossible,
    RoundRobin,
    PPO,  # Reinforcement Learning
    SAC,  # Reinforcement Learning
]
```

Available (commented out):
- **RL**: A2C, DDPG, TD3
- **MPC**: OCMF_V2G, OCMF_G2V, eMPC_V2G, eMPC_G2V
- **Optimization**: V2GProfitMaxOracle, V2GProfitMaxOracleGB, PowerTrackingErrorrMin

### Algorithm Execution Patterns

#### RL Algorithms
```python
# Register Gym environment
gym.envs.register(id='evs-v0', entry_point='ev2gym.ev_city:ev2gym', ...)
env = gym.make('evs-v0')

# Load pre-trained model
model = algorithm.load(load_path, env, device=device)

# Predict actions
action, _ = model.predict(state, deterministic=True)
state, reward, done, stats = env.step(action)
```

#### Heuristic/MPC Algorithms
```python
# Create environment
env = ev2gym_env.EV2Gym(
    config_file=args.config_file,
    load_from_replay_path=replay_path,
    ...
)

# Initialize algorithm
model = algorithm(env=env, replay_path=replay_path)

# Get actions
actions = model.get_action(env=env)
new_state, reward, done, _, stats = env.step(actions)
```

---

## Troubleshooting

### Common Issues

#### Issue: "No replay files found"
**Cause**: Replay files don't exist for the scenario  
**Solution**: The evaluator will automatically generate them, or you can pre-generate:
```bash
python generate_replays.py \
    --config_file ev2gym/example_config_files/simplePST.yaml \
    --n_replays 20
```

#### Issue: "Model not found"
**Cause**: RL model doesn't exist or path is incorrect  
**Solution**: 
1. Check that you trained the model first
2. Verify the model path format: `saved_models/{cs}cs_{scenario}/{algo}_{reward}_{state}/`
3. Ensure reward/state functions match training

#### Issue: "Unknown reward/state function"
**Cause**: Function name not recognized  
**Solution**:
- For built-in: Use exact names (see [Built-in Functions](#built-in-functions))
- For custom: Use `module:function` syntax (see [Custom Functions](#custom-functions))

#### Issue: "Could not import module"
**Cause**: Custom module not found  
**Solution**:
1. Ensure the module file exists in your working directory or PYTHONPATH
2. Check the module path syntax: `my_module:function` or `path.to.module:function`
3. Verify the function exists in the module

#### Issue: Wrong scenario detected / Functions mismatch
**Cause**: Using wrong reward/state functions  
**Solution**: Check your model directory name and specify functions explicitly:
```bash
# Model: ppo_profit_maximization_V2G_profit_max
python evaluator.py \
    --config_file my_config.yaml \
    --reward_function profit_maximization \
    --state_function V2G_profit_max
```

### Performance Considerations

#### GPU Acceleration
- Automatically detects CUDA availability
- RL models load to GPU if available
- Significant speedup for large-scale evaluations

#### Execution Time
Typical execution times per test cycle:
- **Heuristics**: < 1 second
- **MPC**: 1-10 seconds (depends on horizon)
- **RL**: 1-5 seconds
- **Gurobi Optimization**: 5-30 seconds

#### Memory Usage
- Each replay file: ~1-10 MB
- Environment states: ~10-50 MB per algorithm
- Total memory scales with number of algorithms and test cycles

---

## Best Practices

1. **Use Replay Files** - Always use replay files for fair comparison across algorithms
2. **Multiple Test Cycles** - Run at least 10-20 test cycles for statistical significance
3. **Match Training** - Use the same reward/state functions that were used during training
4. **Save Configuration** - The evaluator automatically copies config files to results
5. **Monitor Progress** - Progress is printed for each step and algorithm
6. **Check Results** - Review `results_grouped.txt` for quick insights

---

## Examples

### Example 1: Evaluate Your Trained Models

```bash
# You have trained PPO and SAC models
# Models are in: saved_models/2cs_simplePST/

python evaluator.py \
    --config_file ev2gym/example_config_files/simplePST.yaml \
    --reward_function profit_maximization \
    --state_function V2G_profit_max \
    --n_test_cycles 20
```

### Example 2: Compare Multiple Algorithms

Edit `evaluator.py` to enable more algorithms:
```python
algorithms = [
    ChargeAsFastAsPossible,
    ChargeAsLateAsPossible,
    RoundRobin,
    PPO,
    SAC,
    A2C,  # Uncomment to add
]
```

Then run:
```bash
python evaluator.py \
    --config_file ev2gym/example_config_files/simplePST.yaml \
    --n_test_cycles 20
```

### Example 3: Custom Functions Evaluation

Create custom functions and evaluate:
```bash
python evaluator.py \
    --config_file my_config.yaml \
    --reward_function my_rewards:advanced_reward \
    --state_function my_states:enhanced_state \
    --n_test_cycles 15
```

---

## What is a Replay File?

A **replay file** is a saved record of a complete simulation episode that captures all stochastic elements:

- **EV arrivals**: When each EV arrives and departs
- **EV characteristics**: Battery capacity, initial SoC, desired SoC
- **External signals**: Electricity prices, power setpoints, grid conditions
- **Random events**: Any other stochastic elements

### Why Use Replay Files?

Replay files enable **fair comparison** by ensuring all algorithms face **identical scenarios**:

```
Without Replay Files:
Algorithm A → Random Scenario 1 → Result A
Algorithm B → Random Scenario 2 → Result B  ❌ Not comparable!

With Replay Files:
Algorithm A → Replay File (Scenario 1) → Result A
Algorithm B → Replay File (Scenario 1) → Result B  ✅ Fair comparison!
```

### Replay File Generation

The evaluator automatically generates replay files if they don't exist. You can also pre-generate them:

```bash
python generate_replays.py \
    --config_file ev2gym/example_config_files/simplePST.yaml \
    --n_replays 20
```

---

## References

- **EV2Gym Environment**: `ev2gym/models/ev2gym_env.py`
- **Baseline Algorithms**: `ev2gym/baselines/`
- **RL Training**: `train_stable_baselines.py`
- **Function Loader**: `ev2gym/utilities/loaders.py`
- **Visualization**: `ev2gym/visuals/evaluator_plot.py`

---

**Last Updated**: 2026-01-21  
**Version**: 1.0 (Consolidated)  
**Maintainer**: EV2Gym Team
