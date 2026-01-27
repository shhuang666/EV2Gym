#!/bin/bash

# Script to inspect EV2Gym episode settings without training
# This shows the case settings for each episode including:
# - Number of EVs
# - Initial and target energy
# - Arriving and departure time
# - Battery capacity
# - Charging power limits
# etc.

# Default values
CONFIG_FILE="ev2gym/example_config_files/simplePST.yaml"
STATE_FUNCTION="ev2gym.rl_agent.state:V2G_profit_max_no_forecast"
REWARD_FUNCTION=""
NUM_EPISODES=5
# VERBOSE="--verbose"
# SHOW_PRICES="--show_prices"
SHOW_TIMELINE="--show_timeline"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --quiet)
            VERBOSE="--quiet"
            shift
            ;;
        --show_prices)
            SHOW_PRICES="--show_prices"
            shift
            ;;
        --show_timeline)
            SHOW_TIMELINE="--show_timeline"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config_file PATH] [--num_episodes N] [--quiet] [--show_prices] [--show_timeline] [--reward_function FUNC] [--state_function FUNC]"
            exit 1
            ;;
    esac
done

echo "Inspecting episode settings..."
echo "Config file: $CONFIG_FILE"
echo "Number of episodes: $NUM_EPISODES"
echo ""

# Build the command with optional arguments
CMD="uv run inspect_episode_settings.py --config_file \"$CONFIG_FILE\" --num_episodes \"$NUM_EPISODES\" $VERBOSE $SHOW_PRICES"

if [[ -n "$REWARD_FUNCTION" ]]; then
    CMD="$CMD --reward_function \"$REWARD_FUNCTION\""
fi

if [[ -n "$STATE_FUNCTION" ]]; then
    CMD="$CMD --state_function \"$STATE_FUNCTION\""
fi

if [[ -n "$NUM_EPISODES" ]]; then
    CMD="$CMD --num_episodes \"$NUM_EPISODES\""
fi

if [[ -n "$VERBOSE" ]]; then
    CMD="$CMD --verbose"
fi

if [[ -n "$SHOW_PRICES" ]]; then
    CMD="$CMD --show_prices"
fi

if [[ -n "$SHOW_TIMELINE" ]]; then
    CMD="$CMD --show_timeline"
fi

eval $CMD
