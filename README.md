# SkyLinkSimulator

A fast simulator for a LEO satellite network with ISLs/GSLs, buffer/queueing model, and routing strategies

## Prerequisites

- Python 3.11 or later
- (Recommended) a virtual environment
- Install dependencies from requirements.txt

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts ctivate
pip install -r requirements.txt
```

## Data

By default, `main.py` expects precomputed data under `data/`.

Generate data:

- Positions: `src/calculators/position_calculator.py`
- Visibilities: `src/calculators/neighbour_calculator.py`, `src/calculators/gs_neighbour_calculator.py`
- Traffic: `src/calculators/data_calculator.py`
- Atmosphere/Radio: `src/calculators/atmospheric_attenuation.py`, `src/calculators/rician.py`

The calculator scripts use `CosmicBeats` (included in the repo at `src/calculators/CosmicBeats/`); the configuration is referenced in the
scripts (e.g., `CosmicBeats/configs/oneweb/config.json`).

### Generate/extend precomputed data

Run calculator scripts from `src/calculators/` so their relative paths resolve correctly:

```bash
cd src/calculators
```

Recommended generation order (dependencies):

1) Satellite positions
```bash
python position_calculator.py
```
or multithreaded:
```bash
python position_calculator_mthread.py --workers 4 --total_timepoints 40320 --start_file_index 0
```

2) Inter-satellite visibility grid (depends on positions)
```bash
python neighbour_calculator.py
```

3) Satellite-to-groundstation visibility (depends on positions)
```bash
python gs_neighbour_calculator.py
```

4) Satellite data generation (depends on positions and population map)
```bash
python data_calculator.py
```

Notes:
- `main.py` consumes data in chunks of 1000 steps per file index (`*_0.h5`, `*_1.h5`, ...).
- To run one full week at 15s resolution (`40320` steps), generate at least indices `0..40` for all required streams:
  - `data/grid/grid_{index}.h5`
  - `data/visibility/groundstation_visibility/satellite_visibility_groundstations_{index}.h5`
  - `data/positions/satellite_positions/satellite_positions_{index}.h5`
  - `data/data_generation/satellite_data_generation_{index}.h5`
- Continuation example (from step 5000 / file index 5):
```bash
python position_calculator_mthread.py --workers 4 --total_timepoints 40320 --start_file_index 5
```
- Single-file example (only file index 16, steps 16000..16999):
```bash
python position_calculator_mthread.py --workers 1 --total_timepoints 17000 --start_file_index 16 --end_file_index 16
```

## Quickstart (Simulation)

```bash
python main.py   --growth_factor 2   --gsl_failures False   --isl_failures False   --max_time_steps 240   --logging False   --seed 0   --repetitions 1
```

### CLI arguments

- `--growth_factor` (float): Scales the data generation rate
- `--gsl_failures` (bool): Simulates GSL failures
- `--isl_failures` (bool): Simulates ISL failures
- `--max_time_steps` (int): Number of time steps
- `--logging` (bool): CSV logging in `logging/` and `results/`
- `--seed` (int): Reproducibility
- `--repetitions` (int): Multiple runs per strategy

## Results & Visualization

- Logs/CSV: `logging/`, `results/`
- Plots: `src/visualisation/` (`time_plot.py`, `parameter_plot.py`, `growth_factors_multiple_runs.py`)
