#!/bin/bash

# Batch training script for running multiple experiments
# This script runs multiple training configurations sequentially

# Ensure the script stops on errors
set -e

echo "=========================================="
echo "Starting Batch Training Experiments"
echo "=========================================="
echo ""

# Example 1: Train with PPO on V2GProfitMax config
echo "Experiment 1: PPO with V2GProfitMax"
uv run train_stable_baselines.py \
    --algorithm ppo \
    --device cuda:0 \
    --train_steps 50000 \
    --config_file ev2gym/example_config_files/V2GProfitMax.yaml \
    --run_name "exp1_"

echo ""
echo "Experiment 1 completed!"
echo ""

# Example 2: Train with SAC on V2GProfitPlusLoads config
echo "Experiment 2: SAC with V2GProfitPlusLoads"
uv run train_stable_baselines.py \
    --algorithm sac \
    --device cuda:0 \
    --train_steps 50000 \
    --config_file ev2gym/example_config_files/V2GProfitPlusLoads.yaml \
    --run_name "exp2_"

echo ""
echo "Experiment 2 completed!"
echo ""

# Example 3: Train with TD3 on PublicPST config
echo "Experiment 3: TD3 with PublicPST"
uv run train_stable_baselines.py \
    --algorithm td3 \
    --device cuda:0 \
    --train_steps 50000 \
    --config_file ev2gym/example_config_files/PublicPST.yaml \
    --run_name "exp3_"

echo ""
echo "Experiment 3 completed!"
echo ""

# Example 4: Train with custom state function (no forecast)
echo "Experiment 4: PPO with V2G_profit_max_no_forecast state function"
uv run train_stable_baselines.py \
    --algorithm ppo \
    --device cuda:0 \
    --train_steps 50000 \
    --config_file ev2gym/example_config_files/V2GProfitMax.yaml \
    --state_function ev2gym.rl_agent.state:V2G_profit_max_no_forecast \
    --run_name "exp4_no_forecast_"

echo ""
echo "Experiment 4 completed!"
echo ""

echo "=========================================="
echo "All Batch Experiments Completed!"
echo "=========================================="
