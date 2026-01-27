#!/bin/bash

# Script for running evaluator.py on Linux
# Usage: ./run_evaluator.sh [options]
# Example: ./run_evaluator.sh --config_file ev2gym/example_config_files/V2GProfitMax.yaml --n_test_cycles 10

# ============================================
# DEFAULT CONFIGURATION
# Modify these values to set your preferred defaults
# ============================================

# Configuration file path
CONFIG_FILE="ev2gym/example_config_files/V2GProfitPlusLoads.yaml"

# Number of test cycles (evaluation episodes)
N_TEST_CYCLES=5

# Reward function (leave empty to use default: profit_maximization)
# Examples: "profit_maximization", "SquaredTrackingErrorReward", "ProfitMax_TrPenalty_UserIncentives"
# Or custom: "my_module:custom_reward"
REWARD_FUNCTION="ProfitMax_TrPenalty_UserIncentives"

# State function (leave empty to use default: V2G_profit_max)
# Examples: "V2G_profit_max", "PublicPST", "V2G_profit_max_loads"
# Or custom: "my_module:custom_state"
STATE_FUNCTION="V2G_profit_max_loads"

# Replay path (leave empty to generate new replays)
# Example: "./replay/10cs_1tr_V2GProfitMax/"
# REPLAY_PATH="replay/eval_2cs_1tr_simplePST_4_algos_5_exp_2026_01_23_288154"

# ============================================

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config_file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --n_test_cycles)
            N_TEST_CYCLES="$2"
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
        --replay_path)
            REPLAY_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_evaluator.sh [options]"
            echo ""
            echo "Options:"
            echo "  --config_file FILE       Config file path (default: ev2gym/example_config_files/simplePST.yaml)"
            echo "  --n_test_cycles N        Number of test cycles/episodes (default: 50)"
            echo "  --reward_function FUNC   Reward function (default: profit_maximization)"
            echo "                           Built-in options:"
            echo "                             - profit_maximization"
            echo "                             - SquaredTrackingErrorReward"
            echo "                             - ProfitMax_TrPenalty_UserIncentives"
            echo "                           Custom: my_module:custom_reward"
            echo "  --state_function FUNC    State function (default: V2G_profit_max)"
            echo "                           Built-in options:"
            echo "                             - V2G_profit_max"
            echo "                             - PublicPST"
            echo "                             - V2G_profit_max_loads"
            echo "                           Custom: my_module:custom_state"
            echo "  --replay_path PATH       Path to directory with replay files (optional)"
            echo "                           If not specified, new replays will be generated"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_evaluator.sh"
            echo "  ./run_evaluator.sh --config_file ev2gym/example_config_files/V2GProfitMax.yaml"
            echo "  ./run_evaluator.sh --n_test_cycles 10"
            echo "  ./run_evaluator.sh --reward_function profit_maximization --state_function V2G_profit_max"
            echo "  ./run_evaluator.sh --reward_function my_rewards:custom_reward --state_function my_states:custom_state"
            echo "  ./run_evaluator.sh --replay_path ./replay/10cs_1tr_V2GProfitMax/"
            echo ""
            echo "Note: If --replay_path is not specified, new replay files will be generated."
            echo "      To use existing replays, specify the directory path with --replay_path."
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
CMD="uv run evaluator.py"

if [ -n "$CONFIG_FILE" ]; then
    CMD="$CMD --config_file $CONFIG_FILE"
fi

if [ -n "$N_TEST_CYCLES" ]; then
    CMD="$CMD --n_test_cycles $N_TEST_CYCLES"
fi

if [ -n "$REWARD_FUNCTION" ]; then
    CMD="$CMD --reward_function $REWARD_FUNCTION"
fi

if [ -n "$STATE_FUNCTION" ]; then
    CMD="$CMD --state_function $STATE_FUNCTION"
fi

if [ -n "$REPLAY_PATH" ]; then
    CMD="$CMD --replay_path $REPLAY_PATH"
fi

# Print the command being executed
echo "=========================================="
echo "Running evaluator with the following configuration:"
echo "Config File: $CONFIG_FILE"
echo "Number of Test Cycles: $N_TEST_CYCLES"
if [ -n "$REWARD_FUNCTION" ]; then
    echo "Reward Function: $REWARD_FUNCTION"
else
    echo "Reward Function: profit_maximization (default)"
fi
if [ -n "$STATE_FUNCTION" ]; then
    echo "State Function: $STATE_FUNCTION"
else
    echo "State Function: V2G_profit_max (default)"
fi
if [ -n "$REPLAY_PATH" ]; then
    echo "Replay Path: $REPLAY_PATH"
else
    echo "Replay Path: Will generate new replays"
fi
echo "=========================================="
echo ""
echo "Executing: $CMD"
echo ""

# Execute the command
eval $CMD
