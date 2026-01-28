# EV2Gym Configuration Files Comparison

This document provides a comprehensive comparison of all YAML configuration files in `ev2gym/example_config_files`. These configuration files define different simulation scenarios for the EV2Gym environment, each optimized for specific use cases and research objectives.

## Overview

Configuration files control all aspects of the EV2Gym simulation, including temporal parameters, network topology, pricing, grid constraints, and EV specifications. Each configuration is designed for specific scenarios ranging from simple power setpoint tracking to complex V2G profit maximization with grid constraints.

**Available Configuration Files:**
1. `PublicPST.yaml` - Public charging with power setpoint tracking
2. `V2GProfitMax.yaml` - V2G profit maximization in workplace scenario
3. `V2GProfitPlusLoads.yaml` - V2G profit with inflexible loads and solar PV (default)
4. `V2Ggrid.yaml` - V2G with full grid simulation
5. `simplePST.yaml` - Simplified power setpoint tracking for testing

---

## Table 1: Quick Configuration Comparison

| **Config File** | **Primary Use Case** | **V2G Enabled** | **Grid Simulation** | **Charging Stations** | **Scenario Type** | **Simulation Length** |
|----------------|---------------------|-----------------|--------------------|--------------------|-------------------|---------------------|
| `PublicPST.yaml` | Power setpoint tracking | ❌ No | ❌ No | 20 | Public | 112 steps (28h) |
| `V2GProfitMax.yaml` | Profit maximization | ✅ Yes | ❌ No | 25 | Workplace | 112 steps (28h) |
| `V2GProfitPlusLoads.yaml` | Profit + grid constraints | ✅ Yes | ❌ No | 25 | Workplace | 112 steps (28h) |
| `V2Ggrid.yaml` | Grid-aware V2G | ✅ Yes | ✅ Yes | 150 | Public | 96 steps (24h) |
| `simplePST.yaml` | Simple testing | ❌ No | ❌ No | 2 | Public | 96 steps (24h) |

---

## Table 2: Simulation Parameters Comparison

| **Parameter** | **PublicPST** | **V2GProfitMax** | **V2GProfitPlusLoads** | **V2Ggrid** | **simplePST** |
|--------------|--------------|-----------------|----------------------|------------|--------------|
| **Timescale** | 15 min | 15 min | 15 min | 15 min | 15 min |
| **Simulation Length** | 112 steps | 112 steps | 112 steps | 96 steps | 96 steps |
| **Total Duration** | 28 hours | 28 hours | 28 hours | 24 hours | 24 hours |
| **Random Day** | ✅ True | ✅ True | ✅ True | ✅ True | ❌ False |
| **Random Hour** | ❌ False | ❌ False | ❌ False | ❌ False | ❌ False |
| **Start Hour** | 5:00 | 5:00 | 5:00 | 5:00 | 5:00 |
| **Simulation Days** | Weekdays | Weekdays | Weekdays | Weekdays | Weekdays |

**Notes**:
- All configs use 15-minute timesteps
- Most use 112 steps (28 hours) to cover more than one day
- Random day enabled for most configs to increase training diversity
- All start at 5:00 AM to capture morning charging patterns

---

## Table 3: EV Scenario and Spawn Behavior

| **Parameter** | **PublicPST** | **V2GProfitMax** | **V2GProfitPlusLoads** | **V2Ggrid** | **simplePST** |
|--------------|--------------|-----------------|----------------------|------------|--------------|
| **Scenario** | Public | Workplace | Workplace | Public | Public |
| **Spawn Multiplier** | 5× | 5× | 5× | 5× | 10× |
| **Heterogeneous EVs** | ✅ True | ✅ True | ✅ True | ❌ False | ❌ False |
| **EV Specs File** | ev_plus_phev.json | v2g_enabled2024.json | v2g_enabled2024.json | v2g_enabled2024.json | ev_plus_phev.json |
| **Min Time of Stay** | 60 min | 180 min | 180 min | 200 min | 60 min |

**Scenario Differences**:
- **Public**: Short, unpredictable charging sessions (shopping, errands)
- **Workplace**: Longer, predictable sessions (8+ hours during workday)
- **Private**: Home charging (not used in these configs)

**Spawn Multiplier**: Higher values increase EV arrival probability. Default is 1, but 5-10× recommended for adequate EV presence.

---

## Table 4: Distribution Network Settings

| **Parameter** | **PublicPST** | **V2GProfitMax** | **V2GProfitPlusLoads** | **V2Ggrid** | **simplePST** |
|--------------|--------------|-----------------|----------------------|------------|--------------|
| **V2G Enabled** | ❌ False | ✅ True | ✅ True | ✅ True | ❌ False |
| **Charging Stations** | 20 | 25 | 25 | 150 | 2 |
| **Transformers** | 1 | 1 | 1 | -1 (auto) | 1 |
| **Ports per CS** | 1 | 1 | 1 | 1 | 1 |
| **Simulate Grid** | ❌ False | ❌ False | ❌ False | ✅ True | ❌ False |
| **Network Topology** | None (random) | None (random) | None (random) | None (random) | None (random) |

**V2G (Vehicle-to-Grid)**:
- When enabled, EVs can discharge back to the grid
- Enables energy arbitrage and grid services
- Requires compatible EV specs and charging stations

**Grid Simulation**:
- Only enabled in `V2Ggrid.yaml`
- Simulates full power flow with voltage constraints
- Uses 34-node or 123-node network models
- Significantly increases computational complexity

---

## Table 5: Charging Station Specifications

| **Parameter** | **PublicPST** | **V2GProfitMax** | **V2GProfitPlusLoads** | **V2Ggrid** | **simplePST** |
|--------------|--------------|-----------------|----------------------|------------|--------------|
| **Max Charge Current** | 16 A | 32 A | 32 A | 32 A | 16 A |
| **Max Discharge Current** | 0 A | -32 A | -32 A | -32 A | 0 A |
| **Voltage** | 400 V | 400 V | 400 V | 400 V | 400 V |
| **Phases** | 3 | 3 | 3 | 3 | 3 |
| **Max Charge Power** | ~11 kW | ~22 kW | ~22 kW | ~22 kW | ~11 kW |
| **Max Discharge Power** | 0 kW | ~22 kW | ~22 kW | ~22 kW | 0 kW |

**Power Calculation**: $P = \sqrt{3} \times V \times I \times \cos(\phi) / 1000$ (for 3-phase)
- 16A @ 400V 3-phase ≈ 11 kW
- 32A @ 400V 3-phase ≈ 22 kW

**Discharge Current**: Negative values indicate discharging capability (V2G)

---

## Table 6: Transformer Settings

| **Parameter** | **PublicPST** | **V2GProfitMax** | **V2GProfitPlusLoads** | **V2Ggrid** | **simplePST** |
|--------------|--------------|-----------------|----------------------|------------|--------------|
| **Max Power** | 100 kW | 100 kW | 100 kW | 200 kW | 100 kW |
| **Total CS Capacity** | 220 kW (20×11) | 550 kW (25×22) | 550 kW (25×22) | 3300 kW (150×22) | 22 kW (2×11) |
| **Oversubscription Ratio** | 2.2× | 5.5× | 5.5× | 16.5× | 0.22× |

**Oversubscription**: Ratio of total charging capacity to transformer capacity
- Higher ratios create more challenging coordination problems
- Realistic for shared infrastructure (not all EVs charge simultaneously)
- `simplePST` is undersubscribed (testing scenario)

---

## Table 7: Power Setpoint Settings

| **Parameter** | **PublicPST** | **V2GProfitMax** | **V2GProfitPlusLoads** | **V2Ggrid** | **simplePST** |
|--------------|--------------|-----------------|----------------------|------------|--------------|
| **Setpoint Enabled** | ✅ True | ❌ False | ❌ False | ✅ True | ✅ True |
| **Flexibility** | ±80% | ±80% | ±80% | ±80% | ±70% |

**Power Setpoint Tracking**:
- When enabled, agent must follow a target power consumption profile
- Flexibility defines acceptable deviation range
- Used for grid services (frequency regulation, demand response)
- Profit-focused configs disable this to maximize arbitrage

**Flexibility Examples**:
- 80%: 100 kW setpoint allows 20-180 kW actual consumption
- 70%: 100 kW setpoint allows 30-170 kW actual consumption

---

## Table 8: Inflexible Loads, Solar, and Demand Response

| **Feature** | **PublicPST** | **V2GProfitMax** | **V2GProfitPlusLoads** | **V2Ggrid** | **simplePST** |
|------------|--------------|-----------------|----------------------|------------|--------------|
| **Inflexible Loads** | ❌ False | ❌ False | ✅ True | ❌ False* | ❌ False |
| **Solar PV** | ❌ False | ❌ False | ✅ True | ❌ False* | ❌ False |
| **Demand Response** | ❌ False | ❌ False | ✅ True | ❌ False* | ❌ False |

*Not compatible with grid simulation (`simulate_grid=True`)

### Inflexible Loads Configuration

| **Parameter** | **V2GProfitPlusLoads** | **Others** |
|--------------|----------------------|-----------|
| **Include** | ✅ True | ❌ False |
| **Capacity Multiplier** | 1.0 | 1.0 |
| **Forecast Mean Error** | 30% | 30% |
| **Forecast Std Error** | 5% | 5% |

### Solar Power Configuration

| **Parameter** | **V2GProfitPlusLoads** | **Others** |
|--------------|----------------------|-----------|
| **Include** | ✅ True | ❌ False |
| **Capacity Multiplier** | 1.0 | 1.0 |
| **Forecast Mean Error** | 20% | 20% |
| **Forecast Std Error** | 5% | 5% |

### Demand Response Configuration

| **Parameter** | **V2GProfitPlusLoads** | **Others** |
|--------------|----------------------|-----------|
| **Include** | ✅ True | ❌ False |
| **Events per Day** | 1 | 1 |
| **Capacity Reduction** | 35% ± 5% | 35% ± 5% |
| **Event Duration** | 60 min | 60 min |
| **Event Start Time** | 12:00 ± 2h | 12:00 ± 2h |
| **Notification Time** | 60 min ahead | 60 min ahead |

**Note**: Only `V2GProfitPlusLoads.yaml` enables these features to create a realistic, complex scenario with multiple grid constraints.

---

## Table 9: Default EV Model Specifications

| **Parameter** | **PublicPST** | **V2GProfitMax** | **V2GProfitPlusLoads** | **V2Ggrid** | **simplePST** |
|--------------|--------------|-----------------|----------------------|------------|--------------|
| **Battery Capacity** | 50 kWh | 50 kWh | 50 kWh | 70 kWh | 50 kWh |
| **Max AC Charge** | 11 kW | 11 kW | 11 kW | 22 kW | 11 kW |
| **Max DC Charge** | 50 kW | 50 kW | 50 kW | 50 kW | 50 kW |
| **Max Discharge** | -11 kW | -11 kW | -11 kW | -22 kW | -11 kW |
| **Charge Efficiency** | 100% | 100% | 100% | 100% | 100% |
| **Discharge Efficiency** | 100% | 100% | 100% | 100% | 100% |
| **Desired Capacity** | 100% | 100% | 100% | 100% | 100% |
| **Min Battery Capacity** | 5 kWh | 5 kWh | 5 kWh | 15 kWh | 5 kWh |
| **Min Emergency Capacity** | 25 kWh | 25 kWh | 25 kWh | 15 kWh | 25 kWh |

**Note**: These are default values used when `heterogeneous_ev_specs=False` or as fallback values. When heterogeneous specs are enabled, actual values are loaded from the EV specs JSON file.

---

## Table 10: Grid Simulation Settings (V2Ggrid only)

| **Parameter** | **Value** | **Description** |
|--------------|-----------|----------------|
| **Simulate Grid** | ✅ True | Enable full power flow simulation |
| **VM (p.u.)** | 1.0 | Voltage magnitude in per-unit |
| **S Base** | 1000 kVA | Base power for per-unit system |
| **Load Multiplier** | 1.0 | Scale factor for all loads |
| **PV Scale** | 80% | PV capacity as % of bus load |
| **Bus Info File** | Nodes_34.csv | 34-node network topology |
| **Branch Info File** | Lines_34.csv | 34-node line parameters |
| **PF Solver** | Laurent | Power flow solver algorithm |

**Power Flow Solvers**:
- **Laurent**: Fast linearized power flow approximation
- **PandaPower**: Full AC power flow (more accurate, slower)

**Network Options**:
- 34-node network (default): Smaller, faster
- 123-node network (commented): Larger, more realistic

---

## Detailed Configuration Descriptions

### 1. PublicPST.yaml - Public Power Setpoint Tracking

**Purpose**: Public charging scenario where the agent must track a power setpoint while managing EV charging demand.

**Key Characteristics**:
- ✅ Power setpoint tracking enabled (±80% flexibility)
- ❌ V2G disabled (charging only)
- ✅ Public scenario (short, unpredictable sessions)
- ✅ 20 charging stations at 11 kW each
- ✅ Heterogeneous EV specifications
- ❌ No inflexible loads, solar, or demand response

**Use Cases**:
- Grid frequency regulation
- Demand response programs
- Public charging infrastructure management
- Research on power tracking algorithms

**Compatible State Functions**: `PublicPST`  
**Compatible Rewards**: `SquaredTrackingErrorReward`

**Typical Episode**:
- Duration: 28 hours (112 × 15 min)
- EVs arrive randomly throughout the day
- Agent must balance charging demand with power setpoint
- No revenue optimization (focus on tracking)

---

### 2. V2GProfitMax.yaml - V2G Profit Maximization

**Purpose**: Workplace V2G scenario focused on profit maximization through energy arbitrage.

**Key Characteristics**:
- ✅ V2G enabled (bidirectional charging)
- ✅ Workplace scenario (long, predictable sessions)
- ✅ 25 charging stations at 22 kW each
- ✅ Heterogeneous V2G-capable EVs
- ❌ Power setpoint tracking disabled
- ❌ No inflexible loads, solar, or demand response

**Use Cases**:
- Energy arbitrage (buy low, sell high)
- Workplace charging optimization
- V2G business model research
- Price-responsive charging strategies

**Compatible State Functions**: `V2G_profit_max`, `V2G_profit_max_no_forecast`  
**Compatible Rewards**: `profit_maximization`

**Typical Episode**:
- Duration: 28 hours (112 × 15 min)
- EVs arrive in morning, depart in evening
- Agent exploits price variations for profit
- Minimum 180-minute stay ensures arbitrage opportunities

**Differences from PublicPST**:
- Workplace vs. Public scenario (longer sessions)
- V2G enabled (can discharge)
- Higher charging power (22 kW vs. 11 kW)
- More charging stations (25 vs. 20)
- Different EV specs file (V2G-capable vehicles)

---

### 3. V2GProfitPlusLoads.yaml - V2G with Grid Constraints (Default)

**Purpose**: Most comprehensive and realistic V2G scenario with inflexible loads, solar PV, and demand response events.

**Key Characteristics**:
- ✅ V2G enabled (bidirectional charging)
- ✅ Workplace scenario (long sessions)
- ✅ 25 charging stations at 22 kW each
- ✅ Inflexible loads included
- ✅ Solar PV generation included
- ✅ Demand response events included
- ❌ Power setpoint tracking disabled

**Use Cases**:
- Realistic V2G deployment scenarios
- Multi-objective optimization (profit + grid constraints)
- Renewable energy integration
- Demand response coordination
- Research on complex grid interactions

**Compatible State Functions**: `V2G_profit_max_loads` (default)  
**Compatible Rewards**: `ProfitMax_TrPenalty_UserIncentives`

**Typical Episode**:
- Duration: 28 hours (112 × 15 min)
- EVs arrive in morning with long stay times
- Agent must balance:
  - Profit maximization (price arbitrage)
  - Transformer capacity constraints
  - Inflexible load variations
  - Solar PV generation patterns
  - Demand response events (1 per day, ~60 min)
  - User satisfaction (charge to desired SoC)

**Differences from V2GProfitMax**:
- Adds inflexible loads (residential/commercial)
- Adds solar PV generation
- Adds demand response events
- More complex state space (62 + 2N features vs. 22 + 2N)
- More realistic and challenging

**This is the default configuration** used in most EV2Gym research and examples.

---

### 4. V2Ggrid.yaml - V2G with Full Grid Simulation

**Purpose**: V2G scenario with detailed power flow simulation including voltage constraints and node-level power tracking.

**Key Characteristics**:
- ✅ V2G enabled (bidirectional charging)
- ✅ Full grid simulation enabled
- ✅ 150 charging stations (distributed across grid nodes)
- ✅ 34-node distribution network
- ✅ Power setpoint tracking enabled
- ✅ Homogeneous EV specs (all EVs identical)
- ❌ Inflexible loads/solar/DR not compatible with grid sim

**Use Cases**:
- Grid-aware V2G optimization
- Voltage regulation studies
- Distribution network impact analysis
- Power flow constraint research
- Scalability testing (150 charging stations)

**Compatible State Functions**: `V2G_grid_state`  
**Compatible Rewards**: Custom grid-aware rewards

**Typical Episode**:
- Duration: 24 hours (96 × 15 min)
- 150 EVs distributed across 34 grid nodes
- Agent must consider:
  - Node-level active/reactive power
  - Voltage constraints at each bus
  - Line capacity limits
  - Power setpoint tracking
  - Grid stability

**Grid Simulation Details**:
- Uses Laurent or PandaPower solver
- 34-node network (can use 123-node)
- Tracks voltage magnitude at each node
- Monitors line flows and losses
- Significantly higher computational cost

**Differences from Other Configs**:
- Only config with grid simulation
- Largest scale (150 charging stations)
- Homogeneous EVs (simplifies grid analysis)
- Higher transformer capacity (200 kW)
- Larger EV batteries (70 kWh default)
- Higher charging power (22 kW)

---

### 5. simplePST.yaml - Simplified Power Setpoint Tracking

**Purpose**: Minimal configuration for testing, debugging, and educational purposes.

**Key Characteristics**:
- ✅ Power setpoint tracking enabled (±70% flexibility)
- ❌ V2G disabled (charging only)
- ✅ Public scenario
- ✅ Only 2 charging stations (minimal)
- ✅ High spawn multiplier (10×) to ensure EV presence
- ❌ Random day disabled (deterministic)
- ❌ Homogeneous EV specs (all identical)

**Use Cases**:
- Algorithm testing and debugging
- Educational demonstrations
- Quick experiments
- Baseline comparisons
- Development and prototyping

**Compatible State Functions**: `PublicPST`  
**Compatible Rewards**: `SquaredTrackingErrorReward`

**Typical Episode**:
- Duration: 24 hours (96 × 15 min)
- Deterministic (same day every episode)
- Only 2 EVs maximum
- Simple, predictable behavior
- Fast execution

**Differences from PublicPST**:
- Much smaller scale (2 vs. 20 stations)
- Deterministic (random_day=False)
- Homogeneous EVs (simpler)
- Shorter simulation (96 vs. 112 steps)
- Tighter setpoint flexibility (70% vs. 80%)
- Higher spawn multiplier (10× vs. 5×)
- Undersubscribed transformer (easy scenario)

---

## Configuration Selection Guide

### Decision Tree

```
Do you need grid simulation (voltage, power flow)?
├─ YES → Use V2Ggrid.yaml
└─ NO → Continue
    │
    Do you need V2G (bidirectional charging)?
    ├─ NO → Do you need simple testing?
    │   ├─ YES → Use simplePST.yaml
    │   └─ NO → Use PublicPST.yaml
    │
    └─ YES → Do you need realistic grid constraints?
        ├─ YES → Use V2GProfitPlusLoads.yaml (RECOMMENDED)
        └─ NO → Use V2GProfitMax.yaml
```

### Recommendations by Use Case

| **Use Case** | **Recommended Config** | **Rationale** |
|-------------|----------------------|---------------|
| **Research Publication** | V2GProfitPlusLoads.yaml | Most realistic, comprehensive |
| **V2G Business Model** | V2GProfitMax.yaml | Focus on profit, simpler |
| **Grid Services** | PublicPST.yaml | Power tracking capability |
| **Distribution Network** | V2Ggrid.yaml | Full power flow simulation |
| **Algorithm Development** | simplePST.yaml | Fast, simple, deterministic |
| **Teaching/Learning** | simplePST.yaml → PublicPST.yaml | Progressive complexity |
| **Demand Response** | V2GProfitPlusLoads.yaml | Includes DR events |
| **Renewable Integration** | V2GProfitPlusLoads.yaml | Includes solar PV |

---

## Customization Guide

### Common Modifications

**Adjust Simulation Duration**:
```yaml
simulation_length: 96  # 24 hours
simulation_length: 112  # 28 hours (default for most)
simulation_length: 192  # 48 hours (multi-day)
```

**Change Scenario Type**:
```yaml
scenario: public      # Short, random sessions
scenario: workplace   # Long, predictable sessions  
scenario: private     # Home charging (overnight)
```

**Scale Network Size**:
```yaml
number_of_charging_stations: 10   # Small
number_of_charging_stations: 25   # Medium (default)
number_of_charging_stations: 50   # Large
```

**Enable/Disable V2G**:
```yaml
v2g_enabled: True   # Bidirectional
v2g_enabled: False  # Charging only

# Also update charging station specs:
max_discharge_current: -32  # V2G enabled
max_discharge_current: 0    # V2G disabled
```

**Adjust Transformer Capacity**:
```yaml
transformer:
  max_power: 50   # Tight constraint
  max_power: 100  # Default
  max_power: 200  # Relaxed constraint
```

**Change EV Diversity**:
```yaml
heterogeneous_ev_specs: True   # Realistic variety
heterogeneous_ev_specs: False  # All EVs identical

# Update specs file accordingly:
ev_specs_file: ./ev2gym/data/ev_specs_v2g_enabled2024.json
ev_specs_file: ./ev2gym/data/ev_specs_ev_plus_phev.json
```

---

## Performance Considerations

### Computational Complexity

| **Config** | **Relative Speed** | **State Dimension** | **Bottlenecks** |
|-----------|-------------------|--------------------|-----------------| 
| simplePST | ⚡⚡⚡⚡⚡ Fastest | ~9 | Minimal |
| PublicPST | ⚡⚡⚡⚡ Fast | ~63 | 20 EVs |
| V2GProfitMax | ⚡⚡⚡ Moderate | ~72 | 25 EVs, forecasts |
| V2GProfitPlusLoads | ⚡⚡ Slow | ~112 | 25 EVs, loads, forecasts |
| V2Ggrid | ⚡ Slowest | Variable | Grid simulation, 150 EVs |

### Memory Usage

- **Minimal** (< 1 GB): simplePST, PublicPST
- **Moderate** (1-2 GB): V2GProfitMax, V2GProfitPlusLoads
- **High** (2-4 GB): V2Ggrid (depends on network size)

### Training Time Estimates

For 1M timesteps on typical hardware:

| **Config** | **Estimated Time** | **Notes** |
|-----------|-------------------|-----------|
| simplePST | ~30 min | Fastest convergence |
| PublicPST | ~1-2 hours | Simple dynamics |
| V2GProfitMax | ~2-4 hours | Price forecasts |
| V2GProfitPlusLoads | ~4-8 hours | Complex state space |
| V2Ggrid | ~12-24 hours | Grid simulation overhead |

*Times vary significantly based on hardware, algorithm, and hyperparameters*

---

## Compatibility Matrix

### State Functions

| **Config** | **Primary State Function** | **Alternative State Functions** |
|-----------|---------------------------|--------------------------------|
| PublicPST | `PublicPST` | `BusinessPSTwithMoreKnowledge` |
| V2GProfitMax | `V2G_profit_max` | `V2G_profit_max_no_forecast` |
| V2GProfitPlusLoads | `V2G_profit_max_loads` | `V2G_profit_max` |
| V2Ggrid | `V2G_grid_state` | None (requires grid sim) |
| simplePST | `PublicPST` | None |

### Reward Functions

| **Config** | **Primary Reward** | **Alternative Rewards** |
|-----------|-------------------|------------------------|
| PublicPST | `SquaredTrackingErrorReward` | Custom tracking rewards |
| V2GProfitMax | `profit_maximization` | Custom profit rewards |
| V2GProfitPlusLoads | `ProfitMax_TrPenalty_UserIncentives` | `profit_maximization` |
| V2Ggrid | Custom grid-aware | Voltage penalty rewards |
| simplePST | `SquaredTrackingErrorReward` | Simple rewards |

---

## Common Issues and Solutions

### Issue 1: No EVs Spawning

**Symptoms**: Environment runs but no EVs arrive

**Solutions**:
```yaml
# Increase spawn multiplier
spawn_multiplier: 10  # Higher = more EVs

# Check scenario matches your expectations
scenario: workplace  # Long sessions
scenario: public     # Short sessions

# Ensure min_time_of_stay is reasonable
ev:
  min_time_of_stay: 60  # At least 1 hour
```

### Issue 2: Transformer Overload

**Symptoms**: Constant constraint violations, poor performance

**Solutions**:
```yaml
# Increase transformer capacity
transformer:
  max_power: 200  # Increase from 100

# Or reduce charging stations
number_of_charging_stations: 10  # Reduce from 25

# Or reduce charging power
charging_station:
  max_charge_current: 16  # Reduce from 32
```

### Issue 3: Grid Simulation Errors

**Symptoms**: Crashes or NaN values with V2Ggrid.yaml

**Solutions**:
```yaml
# Try different solver
pf_solver: 'PandaPower'  # Instead of 'Laurent'

# Reduce load multiplier
network_info:
  load_multiplier: 0.5  # Reduce from 1.0

# Use smaller network
bus_info_file: './ev2gym/data/network_data/node_34/Nodes_34.csv'
# Instead of node_123
```

### Issue 4: Training Too Slow

**Symptoms**: Training takes too long

**Solutions**:
```yaml
# Reduce simulation length
simulation_length: 96  # Instead of 112

# Reduce charging stations
number_of_charging_stations: 10  # Instead of 25

# Disable grid simulation
simulate_grid: False

# Disable loads/solar/DR
inflexible_loads:
  include: False
solar_power:
  include: False
demand_response:
  include: False
```

---

## Summary

This document compared all five configuration files in `ev2gym/example_config_files`:

1. **PublicPST.yaml** - Public charging with power setpoint tracking (20 stations, no V2G)
2. **V2GProfitMax.yaml** - Workplace V2G profit maximization (25 stations, simple)
3. **V2GProfitPlusLoads.yaml** - Comprehensive V2G with loads/solar/DR (25 stations, **default**)
4. **V2Ggrid.yaml** - Grid-aware V2G with power flow simulation (150 stations, complex)
5. **simplePST.yaml** - Minimal testing configuration (2 stations, deterministic)

### Quick Selection Guide:

- **Most realistic**: V2GProfitPlusLoads.yaml ⭐ (default)
- **Fastest/simplest**: simplePST.yaml
- **Grid research**: V2Ggrid.yaml
- **V2G basics**: V2GProfitMax.yaml
- **Power tracking**: PublicPST.yaml

For most research and applications, **V2GProfitPlusLoads.yaml** provides the best balance of realism and computational feasibility.
