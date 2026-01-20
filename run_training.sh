#!/bin/bash

# Script for running train_stable_baselines.py on Linux
# Usage: ./run_training.sh [options]
# Example: ./run_training.sh --algorithm ppo --train_steps 50000 --device cuda:0

# ============================================
# DEFAULT CONFIGURATION
# Modify these values to set your preferred defaults
# ============================================

# Algorithm to use: ppo, a2c, ddpg, sac, td3
ALGORITHM="ppo"

# Device to use: cuda:0, cuda:1, cpu, etc.
DEVICE="cpu"

# Number of training steps
TRAIN_STEPS=20000

# Custom run name (leave empty for auto-generated name)
RUN_NAME=""

# Configuration file path
CONFIG_FILE="ev2gym/example_config_files/simplePST.yaml"

# Reward function (leave empty to use from config file)
# Examples: "profit_maximization", "my_module:custom_reward"
REWARD_FUNCTION=""

# State function (leave empty to use from config file)
# Examples: "V2G_profit_max", "my_module:custom_state"
STATE_FUNCTION="ev2gym.rl_agent.state:V2G_profit_max_no_forecast"

# Disable wandb logging (set to "true" to disable, "false" to enable)
NO_WANDB="false"

# ============================================

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --train_steps)
            TRAIN_STEPS="$2"
            shift 2
            ;;
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --config_file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --reward_function)
            REWARD_FUNCTION="$2"
            shift 2
            ;;
        --state_function)
            STATE_FUNCTION="$2"
            shift 2
            ;;
        --no_wandb)
            NO_WANDB="true"
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_training.sh [options]"
            echo ""
            echo "Options:"
            echo "  --algorithm ALGO          Algorithm to use (default: ppo)"
            echo "                           Options: ppo, a2c, ddpg, sac, td3"
            echo "  --device DEVICE          Device to use (default: cuda:0)"
            echo "  --train_steps STEPS      Number of training steps (default: 20000)"
            echo "  --run_name NAME          Name for the run (default: auto-generated)"
            echo "  --config_file FILE       Config file path (default: ev2gym/example_config_files/V2GProfitPlusLoads.yaml)"
            echo "  --reward_function FUNC   Reward function (default: from config)"
            echo "                           Examples: profit_maximization, my_module:custom_reward"
            echo "  --state_function FUNC    State function (default: from config)"
            echo "                           Examples: V2G_profit_max, my_module:custom_state"
            echo "  --no_wandb              Disable wandb logging (default: enabled)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_training.sh --algorithm ppo --train_steps 50000"
            echo "  ./run_training.sh --config_file ev2gym/example_config_files/V2GProfitMax.yaml"
            echo "  ./run_training.sh --reward_function profit_maximization --state_function V2G_profit_max_no_forecast"
            echo "  ./run_training.sh --reward_function my_rewards:custom_reward --state_function my_states:custom_state"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build the command
CMD="uv run train_stable_baselines.py --algorithm $ALGORITHM --device $DEVICE --train_steps $TRAIN_STEPS --config_file $CONFIG_FILE"

if [ -n "$RUN_NAME" ]; then
    CMD="$CMD --run_name $RUN_NAME"
fi

if [ -n "$REWARD_FUNCTION" ]; then
    CMD="$CMD --reward_function $REWARD_FUNCTION"
fi

if [ -n "$STATE_FUNCTION" ]; then
    CMD="$CMD --state_function $STATE_FUNCTION"
fi

if [ "$NO_WANDB" = "true" ]; then
    CMD="$CMD --no_wandb"
fi

# Print the command being executed
echo "=========================================="
echo "Running training with the following configuration:"
echo "Algorithm: $ALGORITHM"
echo "Device: $DEVICE"
echo "Training Steps: $TRAIN_STEPS"
echo "Config File: $CONFIG_FILE"
if [ -n "$RUN_NAME" ]; then
    echo "Run Name: $RUN_NAME"
fi
if [ -n "$REWARD_FUNCTION" ]; then
    echo "Reward Function: $REWARD_FUNCTION"
fi
if [ -n "$STATE_FUNCTION" ]; then
    echo "State Function: $STATE_FUNCTION"
fi
if [ "$NO_WANDB" = "true" ]; then
    echo "Wandb: Disabled"
else
    echo "Wandb: Enabled"
fi
echo "=========================================="
echo ""
echo "Executing: $CMD"
echo ""

# Execute the command
eval $CMD
