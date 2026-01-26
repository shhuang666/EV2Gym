# Plotting Files Clarification

## Summary

You're correct that there's overlap between `plots.py` and `evaluator_plot.py`, but they serve **different purposes** and are used in **different contexts**.

## File Purposes

### 1. `ev2gym/visuals/plots.py`
- **Purpose**: Used for **single simulation** visualization
- **Main Function**: `ev_city_plot(env)`
- **When Used**: Called at the end of a **single** simulation when `save_plots=True` is set in the environment
- **Where Called**: In `ev2gym_env.py` line 490, inside the `_check_termination()` method
- **What It Plots**:
  - Total power and current of each transformer
  - Current of each charging station
  - Energy level of each EV in charging stations
  - Total power of the CPO
  - Detailed single-environment metrics

### 2. `ev2gym/visuals/evaluator_plot.py`
- **Purpose**: Used for **comparative evaluation** across multiple algorithms
- **Main Functions**: 
  - `plot_total_power()`
  - `plot_total_power_V2G()`
  - `plot_comparable_EV_SoC()`
  - `plot_comparable_EV_SoC_single()`
  - `plot_comparable_CS_Power()`
  - `plot_actual_power_vs_setpoint()`
  - `plot_prices()`
- **When Used**: Called by `evaluator.py` after running multiple algorithms
- **Where Called**: In `evaluator.py` lines 401-435
- **What It Plots**:
  - **Comparative plots** showing multiple algorithms side-by-side
  - Designed to compare performance across different control strategies
  - Takes a dictionary of results from multiple algorithm runs

## Usage in `evaluator.py`

**YES**, when running `evaluator.py`, **ONLY** the functions from `evaluator_plot.py` are used, **NOT** `ev_city_plot` from `plots.py`.

Here's what happens in `evaluator.py`:

```python
# Lines 44-49: Import only from evaluator_plot.py
from ev2gym.visuals.evaluator_plot import plot_total_power, plot_comparable_EV_SoC
from ev2gym.visuals.evaluator_plot import (
    plot_total_power_V2G,
    plot_actual_power_vs_setpoint,
)
from ev2gym.visuals.evaluator_plot import plot_comparable_EV_SoC_single, plot_prices

# Lines 401-435: Call the comparative plotting functions
plot_total_power(results_path=save_path + "plot_results_dict.pkl", ...)
plot_comparable_EV_SoC(results_path=save_path + "plot_results_dict.pkl", ...)
plot_actual_power_vs_setpoint(results_path=save_path + "plot_results_dict.pkl", ...)
plot_total_power_V2G(results_path=save_path + "plot_results_dict.pkl", ...)
plot_comparable_EV_SoC_single(results_path=save_path + "plot_results_dict.pkl", ...)
plot_prices(results_path=save_path + "plot_results_dict.pkl", ...)
```

## Key Differences

| Aspect | `plots.py` (`ev_city_plot`) | `evaluator_plot.py` |
|--------|----------------------------|---------------------|
| **Input** | Single `env` object | Dictionary of multiple environments |
| **Purpose** | Visualize one simulation | Compare multiple algorithms |
| **Called By** | `EV2Gym` environment itself | `evaluator.py` script |
| **Trigger** | `save_plots=True` in env | Running evaluator script |
| **Plot Type** | Detailed single-run plots | Comparative multi-algorithm plots |

## Common Functionality

Yes, there is overlap in **what** they plot (power, EV energy, etc.), but:
- `ev_city_plot` shows these metrics for **one simulation**
- `evaluator_plot.py` functions show these metrics for **multiple algorithms simultaneously** for comparison

## Recommendation

The code structure makes sense for its use case:
- If you're running a **single simulation** (e.g., `example.py`), use `plots.py`
- If you're **comparing algorithms** (e.g., `evaluator.py`), use `evaluator_plot.py`

However, there could be some refactoring opportunities to reduce code duplication by:
1. Creating shared helper functions for common plotting logic
2. Having `evaluator_plot.py` call `plots.py` functions internally for individual algorithm plots
3. Consolidating similar plotting code into a shared utilities module

But as it stands, both files will **not** be used simultaneously when running `evaluator.py` - only `evaluator_plot.py` functions are called.
