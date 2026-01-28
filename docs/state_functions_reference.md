# EV2Gym State Functions Reference

This document provides a comprehensive reference for all state functions defined in `ev2gym/rl_agent/state.py`. State functions determine what information the RL agent observes from the environment at each timestep.

## Overview

State functions transform the environment's internal state into observation vectors that the RL agent uses for decision-making. Each state function is designed for specific scenarios and includes different combinations of temporal, economic, grid, and EV-related features.

**Available State Functions:**
1. `PublicPST` - Public charging with power setpoint tracking
2. `V2G_profit_max` - V2G profit maximization with price forecasts
3. `V2G_profit_max_loads` - V2G profit maximization with loads and grid constraints
4. `BusinessPSTwithMoreKnowledge` - Business scenario with detailed EV information
5. `V2G_grid_state` - V2G with grid state information
6. `V2G_profit_max_no_forecast` - Simplified V2G without price forecasts

---

## Table 1: State Functions Quick Comparison

| **Function Name** | **Primary Use Case** | **State Dimension** | **Forecast Horizon** | **Grid Info** | **EV Features** |
|-------------------|---------------------|---------------------|---------------------|---------------|-----------------|
| `PublicPST` | Power setpoint tracking | $3 + 3N_{cs}$ | None | Power setpoint | Full flag, energy, time since arrival |
| `V2G_profit_max` | V2G profit maximization | $22 + 2N_{cs}$ | 20 steps (prices) | None | SoC, time to departure |
| `V2G_profit_max_loads` | V2G with grid constraints | $62 + 2N_{cs}$ | 20 steps (prices, loads, limits) | Loads, PV, limits | SoC, time to departure |
| `BusinessPSTwithMoreKnowledge` | Business scenario | $3 + 3N_{cs}$ | None | Power setpoint, potential | SoC, arrival, departure times |
| `V2G_grid_state` | V2G with grid simulation | $5 + 2N_{nodes} + 3N_{cs}$ | 1 step (prices) | Node powers, setpoint | Capacity, departure, bus ID |
| `V2G_profit_max_no_forecast` | Simplified V2G | $2 + N_{prices} + 2N_{cs}$ | None | None | SoC, time to departure |

*Where $N_{cs}$ = number of charging stations, $N_{nodes}$ = number of grid nodes, $N_{prices}$ = total simulation length*

---

## Detailed State Function Descriptions

### 1. `PublicPST` - Public Power Setpoint Tracking

**Purpose**: Designed for public charging scenarios where the agent must track a power setpoint while managing EV charging.

**Code Reference**: Lines 6-63

**State Vector Components**:
```python
state = [
    current_step / simulation_length,      # 1 feature (normalized time)
    power_setpoint[t],                     # 1 feature (target power)
    current_power_usage[t-1],              # 1 feature (previous power usage)
    # For each charging station:
    [full_flag, energy_exchanged, time_since_arrival]  # 3 features × N_cs
]
```

**Total Dimension**: $1 + 1 + 1 + 3 \times N_{cs} = 3 + 3N_{cs}$

For 20 charging stations: **63 features**

**Feature Details**:

| **Feature** | **Type** | **Range/Units** | **Description** |
|-------------|----------|-----------------|-----------------|
| Normalized step | Temporal | [0, 1] | Current timestep divided by total simulation length |
| Power setpoint | Grid | kW | Target power consumption for current step |
| Current power usage | Grid | kW | Actual power consumption from previous step |
| Full flag | EV | {0.5, 1.0} | 1.0 if EV is fully charged, 0.5 otherwise |
| Energy exchanged | EV | kWh | Total energy exchanged with EV since arrival |
| Time since arrival | EV | timesteps | Number of steps since EV arrived |

**Key Characteristics**:
- ✅ Normalized temporal information (0-1 scale)
- ✅ Power setpoint tracking capability
- ✅ Binary full charge detection
- ❌ No price information
- ❌ No future forecasts
- ❌ No departure time information

**Best Used With**:
- Configuration: `PublicPST.yaml`
- Reward: `SquaredTrackingErrorReward`
- Scenario: Public charging with grid service requirements

---

### 2. `V2G_profit_max` - V2G Profit Maximization

**Purpose**: Designed for V2G scenarios focused on profit maximization through energy arbitrage using price forecasts.

**Code Reference**: Lines 65-106

**State Vector Components**:
```python
state = [
    current_step,                          # 1 feature (absolute time)
    current_power_usage[t-1],              # 1 feature (previous power)
    charge_prices[t:t+20],                 # 20 features (price forecast)
    # For each charging station:
    [EV_SoC, time_to_departure]            # 2 features × N_cs
]
```

**Total Dimension**: $1 + 1 + 20 + 2 \times N_{cs} = 22 + 2N_{cs}$

For 25 charging stations: **72 features**

**Feature Details**:

| **Feature** | **Type** | **Range/Units** | **Description** |
|-------------|----------|-----------------|-----------------|
| Current step | Temporal | [0, simulation_length] | Absolute timestep counter |
| Current power usage | Grid | kW | Actual power consumption from previous step |
| Charge prices (×20) | Economic | $/kWh (absolute value) | Price forecast for next 20 timesteps |
| EV SoC | EV | [0, 1] | State of charge (battery percentage) |
| Time to departure | EV | timesteps | Remaining time until EV departs |

**Key Characteristics**:
- ✅ 20-step price forecast for arbitrage opportunities
- ✅ Absolute timestep information
- ✅ Simple EV state (SoC + departure)
- ✅ Zero-padding for forecast beyond simulation end
- ❌ No grid constraint information
- ❌ No external load information

**Best Used With**:
- Configuration: `V2GProfitMax.yaml`
- Reward: `profit_maximization`
- Scenario: Workplace V2G with profit focus

**Implementation Notes**:
- Price forecast uses `abs()` to ensure positive values
- Forecasts shorter than 20 steps are zero-padded
- Time to departure can be negative if EV overstays

---

### 3. `V2G_profit_max_loads` - V2G with Grid Constraints

**Purpose**: Most comprehensive state function for V2G scenarios with external loads, solar PV, and transformer constraints.

**Code Reference**: Lines 108-155

**State Vector Components**:
```python
state = [
    current_step,                          # 1 feature (absolute time)
    current_power_usage[t-1],              # 1 feature (previous power)
    charge_prices[t:t+20],                 # 20 features (price forecast)
    loads_minus_pv[t:t+20],                # 20 features (net load forecast)
    power_limits[t:t+20],                  # 20 features (transformer limits)
    # For each charging station:
    [EV_SoC, time_to_departure]            # 2 features × N_cs
]
```

**Total Dimension**: $1 + 1 + 20 + 20 + 20 + 2 \times N_{cs} = 62 + 2N_{cs}$

For 25 charging stations: **112 features**

**Feature Details**:

| **Feature** | **Type** | **Range/Units** | **Description** |
|-------------|----------|-----------------|-----------------|
| Current step | Temporal | [0, simulation_length] | Absolute timestep counter |
| Current power usage | Grid | kW | Actual power consumption from previous step |
| Charge prices (×20) | Economic | $/kWh (absolute value) | Price forecast for next 20 timesteps |
| Net loads (×20) | Grid | kW | Forecast of loads minus PV generation |
| Power limits (×20) | Grid | kW | Transformer power capacity limits |
| EV SoC | EV | [0, 1] | State of charge (battery percentage) |
| Time to departure | EV | timesteps | Remaining time until EV departs |

**Key Characteristics**:
- ✅ Comprehensive 20-step forecasts (prices, loads, limits)
- ✅ Net load information (loads - PV)
- ✅ Transformer capacity constraints
- ✅ Suitable for demand response scenarios
- ✅ Accounts for renewable generation
- ✅ Highest state dimensionality

**Best Used With**:
- Configuration: `V2GProfitPlusLoads.yaml` (default)
- Reward: `ProfitMax_TrPenalty_UserIncentives`
- Scenario: Real-world deployments with grid constraints

**Implementation Notes**:
- Uses `tr.get_load_pv_forecast()` for net load prediction
- Uses `tr.get_power_limits()` for capacity constraints
- Designed for scenarios with inflexible loads and solar PV
- Most realistic representation of grid-connected systems

---

### 4. `BusinessPSTwithMoreKnowledge` - Business Scenario with Detailed EV Info

**Purpose**: Business/workplace scenario requiring detailed EV information including arrival and departure times for advanced scheduling.

**Code Reference**: Lines 157-211

**State Vector Components**:
```python
state = [
    current_step / simulation_length,      # 1 feature (normalized time)
    power_setpoint[t],                     # 1 feature (target power)
    charge_power_potential[t],             # 1 feature (max available power)
    transformer_max_current / 100,         # 1 feature (scaled current limit)
    # For each charging station:
    [time_of_arrival, expected_departure, SoC]  # 3 features × N_cs
]
```

**Total Dimension**: $1 + 1 + 1 + 1 + 3 \times N_{cs} = 4 + 3N_{cs}$

For 25 charging stations: **79 features**

**Feature Details**:

| **Feature** | **Type** | **Range/Units** | **Description** |
|-------------|----------|-----------------|-----------------|
| Normalized step | Temporal | [0, 1] | Current timestep divided by total simulation length |
| Power setpoint | Grid | kW | Target power consumption for current step |
| Charge power potential | Grid | kW | Maximum available charging power |
| Transformer max current | Grid | A (scaled ÷100) | Transformer current capacity |
| Time of arrival | EV | [0, 1] (normalized) | When EV arrived (normalized) |
| Expected departure | EV | [0, 1] (normalized) | When EV expects to depart (normalized) |
| EV SoC | EV | [0, 1] | State of charge (battery percentage) |

**Key Characteristics**:
- ✅ Detailed temporal EV information (arrival + departure)
- ✅ Power setpoint and potential tracking
- ✅ Transformer current limit awareness
- ✅ All temporal features normalized to [0, 1]
- ❌ No price information
- ❌ No forecasts

**Best Used With**:
- Business/workplace scenarios with power setpoint requirements
- Scenarios requiring detailed EV scheduling
- Cases where arrival and departure patterns are important

**Implementation Notes**:
- Uses `etime_of_departure` (expected departure time)
- All temporal features normalized by `simulation_length`
- Transformer max current scaled down by factor of 100
- Contains commented-out features for additional EV metrics

---

### 5. `V2G_grid_state` - V2G with Grid Simulation

**Purpose**: V2G scenarios with detailed grid state information including node-level active and reactive power.

**Code Reference**: Lines 213-275

**State Vector Components**:
```python
state = [
    weekday / 7,                           # 1 feature (day of week)
    sin(hour / 24 * 2π),                   # 1 feature (hour sine)
    cos(hour / 24 * 2π),                   # 1 feature (hour cosine)
    charge_prices[t:t+1],                  # 1 feature (next price)
    power_setpoint[t],                     # 1 feature (target power)
    current_power_usage[t-1],              # 1 feature (previous power)
    node_active_power[1:, t-1],            # N_nodes features (active power)
    node_reactive_power[1:, t-1],          # N_nodes features (reactive power)
    # For each charging station:
    [current_capacity, time_to_departure, connected_bus]  # 3 features × N_cs
]
```

**Total Dimension**: $1 + 1 + 1 + 1 + 1 + 1 + N_{nodes} + N_{nodes} + 3 \times N_{cs} = 6 + 2N_{nodes} + 3N_{cs}$

**Feature Details**:

| **Feature** | **Type** | **Range/Units** | **Description** |
|-------------|----------|-----------------|-----------------|
| Weekday | Temporal | [0, 1] | Day of week normalized (0-6 → 0-1) |
| Hour (sin) | Temporal | [-1, 1] | Sine encoding of hour for cyclical pattern |
| Hour (cos) | Temporal | [-1, 1] | Cosine encoding of hour for cyclical pattern |
| Charge price | Economic | $/kWh | Price for next timestep only |
| Power setpoint | Grid | kW | Target power consumption |
| Current power usage | Grid | kW | Actual power from previous step |
| Node active power | Grid | kW | Active power at each grid node |
| Node reactive power | Grid | kVAR | Reactive power at each grid node |
| Current capacity | EV | kWh | Current battery energy content |
| Time to departure | EV | timesteps | Remaining time until departure (+1) |
| Connected bus | EV | bus ID | Grid bus/node where EV is connected |

**Key Characteristics**:
- ✅ Cyclical temporal encoding (sin/cos for hour)
- ✅ Day-of-week awareness
- ✅ Detailed grid state (all node powers)
- ✅ Bus/node location information for each EV
- ✅ Designed for grid simulation scenarios
- ❌ Only 1-step price forecast
- ❌ Uses current capacity instead of SoC

**Best Used With**:
- Scenarios with `simulate_grid=True`
- Grid-aware optimization problems
- Cases requiring voltage/power flow considerations

**Implementation Notes**:
- Requires grid simulation to be enabled
- Uses `env.node_active_power` and `env.node_reactive_power`
- Skips first node (index 0) in node power arrays
- Time to departure has +1 offset
- Uses absolute battery capacity instead of normalized SoC

---

### 6. `V2G_profit_max_no_forecast` - Simplified V2G

**Purpose**: Simplified version of V2G profit maximization without price forecasting, using all prices at once.

**Code Reference**: Lines 277-314

**State Vector Components**:
```python
state = [
    current_step,                          # 1 feature (absolute time)
    current_power_usage[t-1],              # 1 feature (previous power)
    charge_prices[all],                    # N_prices features (all prices)
    # For each charging station:
    [EV_SoC, time_to_departure]            # 2 features × N_cs
]
```

**Total Dimension**: $1 + 1 + N_{prices} + 2 \times N_{cs} = 2 + N_{prices} + 2N_{cs}$

For 112-step simulation with 25 charging stations: **164 features**

**Feature Details**:

| **Feature** | **Type** | **Range/Units** | **Description** |
|-------------|----------|-----------------|-----------------|
| Current step | Temporal | [0, simulation_length] | Absolute timestep counter |
| Current power usage | Grid | kW | Actual power consumption from previous step |
| Charge prices (all) | Economic | $/kWh (absolute value) | All prices for entire simulation |
| EV SoC | EV | [0, 1] | State of charge (battery percentage) |
| Time to departure | EV | timesteps | Remaining time until EV departs |

**Key Characteristics**:
- ✅ Complete price information (all timesteps)
- ✅ Simple EV state representation
- ✅ No forecast horizon limitation
- ⚠️ Very high dimensionality due to all prices
- ❌ No grid constraint information
- ❌ Less efficient than windowed forecasts

**Best Used With**:
- Experimental scenarios
- Cases where agent needs full price visibility
- Comparison studies against forecast-based approaches

**Implementation Notes**:
- Uses `env.charge_prices[0]` to get all prices
- No zero-padding needed (all prices available)
- State dimension grows with simulation length
- May be less sample-efficient than windowed forecasts

---

## Table 2: Feature Category Breakdown

| **State Function** | **Temporal** | **Economic** | **Grid/Power** | **EV Info** | **Total (25 CS)** |
|-------------------|--------------|--------------|----------------|-------------|-------------------|
| `PublicPST` | 1 (normalized) | 0 | 2 (setpoint, usage) | 75 (3×25) | 78* |
| `V2G_profit_max` | 1 (absolute) | 20 (forecast) | 1 (usage) | 50 (2×25) | 72 |
| `V2G_profit_max_loads` | 1 (absolute) | 20 (forecast) | 41 (usage + loads + limits) | 50 (2×25) | 112 |
| `BusinessPSTwithMoreKnowledge` | 1 (normalized) | 0 | 3 (setpoint, potential, tr limit) | 75 (3×25) | 79 |
| `V2G_grid_state` | 3 (day, hour) | 1 (next step) | 2N+2 (nodes + setpoint + usage) | 75 (3×25) | Variable** |
| `V2G_profit_max_no_forecast` | 1 (absolute) | 112 (all prices) | 1 (usage) | 50 (2×25) | 164*** |

*PublicPST typically uses 20 charging stations (63 features)  
**Depends on number of grid nodes  
***Assumes 112-step simulation

---

## Table 3: Forecast Horizons Comparison

| **State Function** | **Price Forecast** | **Load Forecast** | **Power Limit Forecast** | **Total Forecast Features** |
|-------------------|-------------------|-------------------|-------------------------|----------------------------|
| `PublicPST` | ❌ None | ❌ None | ❌ None | 0 |
| `V2G_profit_max` | ✅ 20 steps | ❌ None | ❌ None | 20 |
| `V2G_profit_max_loads` | ✅ 20 steps | ✅ 20 steps | ✅ 20 steps | 60 |
| `BusinessPSTwithMoreKnowledge` | ❌ None | ❌ None | ❌ None | 0 |
| `V2G_grid_state` | ⚠️ 1 step | ❌ None | ❌ None | 1 |
| `V2G_profit_max_no_forecast` | ⚠️ All steps | ❌ None | ❌ None | 112* |

*Entire price array, not a rolling forecast

---

## Table 4: EV Feature Comparison

| **State Function** | **Features per EV** | **Feature Names** | **Information Type** |
|-------------------|---------------------|-------------------|---------------------|
| `PublicPST` | 3 | Full flag, Energy exchanged, Time since arrival | Charging status & history |
| `V2G_profit_max` | 2 | SoC, Time to departure | Current state & urgency |
| `V2G_profit_max_loads` | 2 | SoC, Time to departure | Current state & urgency |
| `BusinessPSTwithMoreKnowledge` | 3 | Time of arrival, Expected departure, SoC | Full temporal profile |
| `V2G_grid_state` | 3 | Current capacity, Time to departure, Connected bus | Grid-aware state |
| `V2G_profit_max_no_forecast` | 2 | SoC, Time to departure | Current state & urgency |

---

## Design Patterns and Best Practices

### Temporal Encoding

**Absolute Timesteps** (`V2G_profit_max`, `V2G_profit_max_loads`, `V2G_profit_max_no_forecast`):
- Uses raw timestep counter
- Suitable when absolute position in episode matters
- Agent learns episode length implicitly

**Normalized Timesteps** (`PublicPST`, `BusinessPSTwithMoreKnowledge`):
- Divides by `simulation_length` to get [0, 1] range
- Better for generalization across different episode lengths
- Explicit progress indicator

**Cyclical Encoding** (`V2G_grid_state`):
- Uses sin/cos for hour-of-day
- Captures cyclical nature of time
- Prevents discontinuity at midnight

### Price Information Strategies

**Rolling Window Forecast** (`V2G_profit_max`, `V2G_profit_max_loads`):
- 20-step lookahead window
- Zero-padded when approaching end
- Balances information and dimensionality

**Full Price Array** (`V2G_profit_max_no_forecast`):
- All prices visible at once
- High dimensionality
- Complete information but harder to learn

**No Prices** (`PublicPST`, `BusinessPSTwithMoreKnowledge`):
- Focus on power tracking or scheduling
- Suitable when prices are not the primary objective

### EV State Representation

**SoC + Time to Departure** (Most common):
- Captures current state and urgency
- Minimal but sufficient for most scenarios
- 2 features per EV

**Full Temporal Profile** (`BusinessPSTwithMoreKnowledge`):
- Arrival, departure, and SoC
- Enables sophisticated scheduling
- 3 features per EV

**Charging History** (`PublicPST`):
- Full flag and energy exchanged
- Useful for tracking-based objectives
- 3 features per EV

### Zero-Padding for Empty Slots

All state functions use zero-padding when no EV is connected:
```python
if EV is not None:
    state.append([feature1, feature2, ...])
else:
    state.append(np.zeros(N_features))
```

This ensures consistent state dimensionality regardless of EV presence.

---

## Usage Guidelines

### Choosing the Right State Function

**For Profit Maximization**:
- Start with `V2G_profit_max` for simple scenarios
- Use `V2G_profit_max_loads` for realistic deployments with grid constraints

**For Power Tracking**:
- Use `PublicPST` for public charging scenarios
- Use `BusinessPSTwithMoreKnowledge` if detailed EV scheduling is needed

**For Grid-Aware Optimization**:
- Use `V2G_grid_state` when grid simulation is enabled
- Requires `simulate_grid=True` in configuration

**For Experimentation**:
- Use `V2G_profit_max_no_forecast` to study impact of forecast horizons
- Compare against windowed forecast approaches

### State Function Selection Checklist

- [ ] Does your scenario involve V2G (bidirectional charging)?
- [ ] Do you need price forecasts for arbitrage?
- [ ] Are there external loads or solar PV?
- [ ] Do you need to track power setpoints?
- [ ] Is grid simulation enabled?
- [ ] What is your primary optimization objective?

---

## Implementation Notes

### Common Patterns

All state functions follow this structure:
1. Initialize state list with temporal features
2. Add grid/power features
3. Add economic features (if applicable)
4. Iterate through transformers and charging stations
5. Add EV-specific features (or zeros if no EV)
6. Flatten and return as numpy array

### State Vector Construction

```python
state = []  # Initialize empty list
state.append(feature1)  # Add scalar
state.append(feature_array)  # Add array
state.append([f1, f2, f3])  # Add list
return np.array(np.hstack(state))  # Flatten and convert
```

### Handling Edge Cases

**End of Simulation**:
- Forecasts are zero-padded when extending beyond simulation length
- Some functions check `if env.current_step < env.simulation_length`

**First Timestep**:
- Previous power usage uses `env.current_step-1` (may be 0)
- `V2G_grid_state` has special handling for step 0

**Missing EVs**:
- All functions use zero-padding for empty charging stations
- Maintains consistent dimensionality

---

## Compatibility Matrix

| **State Function** | **Compatible Configs** | **Compatible Rewards** | **Requires Grid Sim** |
|-------------------|----------------------|----------------------|---------------------|
| `PublicPST` | PublicPST.yaml | SquaredTrackingErrorReward | ❌ No |
| `V2G_profit_max` | V2GProfitMax.yaml | profit_maximization | ❌ No |
| `V2G_profit_max_loads` | V2GProfitPlusLoads.yaml | ProfitMax_TrPenalty_UserIncentives | ❌ No |
| `BusinessPSTwithMoreKnowledge` | Custom business configs | Custom rewards | ❌ No |
| `V2G_grid_state` | Configs with grid simulation | Custom grid-aware rewards | ✅ Yes |
| `V2G_profit_max_no_forecast` | V2GProfitMax.yaml | profit_maximization | ❌ No |

---

## Performance Considerations

### State Dimensionality Impact

Higher-dimensional states require:
- More network parameters in the policy/value networks
- More samples for effective learning
- Longer training times

**Recommendations**:
- Use smallest state that captures necessary information
- Prefer windowed forecasts over full arrays
- Consider feature engineering to reduce dimensionality

### Computational Efficiency

**Most Efficient**: `PublicPST`, `V2G_profit_max`
- Minimal feature computation
- No complex forecasts

**Moderate**: `V2G_profit_max_loads`, `BusinessPSTwithMoreKnowledge`
- Additional forecast computations
- Transformer queries for loads/limits

**Least Efficient**: `V2G_grid_state`
- Grid simulation overhead
- Node-level power calculations

---

## Custom State Function Development

### Template

```python
def custom_state_function(env, *args):
    '''
    Description of your state function
    '''
    state = []
    
    # 1. Add temporal features
    state.append(env.current_step)
    
    # 2. Add grid/power features
    state.append(env.current_power_usage[env.current_step-1])
    
    # 3. Add economic features (optional)
    # state.append(env.charge_prices[...])
    
    # 4. Add EV features
    for tr in env.transformers:
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:
                        state.append([
                            # Your EV features here
                        ])
                    else:
                        state.append(np.zeros(N_features))
    
    return np.array(np.hstack(state))
```

### Best Practices

1. **Document your state function** with clear docstrings
2. **Use consistent zero-padding** for empty charging stations
3. **Normalize features** when possible (especially temporal)
4. **Test state dimensionality** matches expectations
5. **Consider forecast horizons** carefully (balance information vs. dimensionality)
6. **Handle edge cases** (first step, last step, empty EVs)

---

## Debugging State Functions

### Common Issues

**Dimension Mismatch**:
```python
# Check state dimension
state = state_function(env)
print(f"State shape: {state.shape}")
print(f"Expected: {expected_dimension}")
```

**NaN or Inf Values**:
```python
# Check for invalid values
assert not np.isnan(state).any(), "State contains NaN"
assert not np.isinf(state).any(), "State contains Inf"
```

**Feature Ranges**:
```python
# Verify feature ranges
print(f"State min: {state.min()}, max: {state.max()}")
print(f"State mean: {state.mean()}, std: {state.std()}")
```

---

## Summary

This reference document covers all six state functions in `ev2gym/rl_agent/state.py`:

1. **PublicPST** - Power setpoint tracking for public charging
2. **V2G_profit_max** - V2G profit with price forecasts
3. **V2G_profit_max_loads** - Comprehensive V2G with grid constraints (default)
4. **BusinessPSTwithMoreKnowledge** - Business scenario with detailed EV info
5. **V2G_grid_state** - Grid-aware V2G with node-level information
6. **V2G_profit_max_no_forecast** - Simplified V2G without forecasts

Choose your state function based on:
- Your optimization objective (profit, tracking, scheduling)
- Available information (prices, loads, grid state)
- Computational constraints (state dimensionality)
- Scenario type (public, workplace, grid-connected)

For most real-world applications, **`V2G_profit_max_loads`** provides the best balance of information richness and practical applicability.
