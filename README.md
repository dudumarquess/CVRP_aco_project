# Running Experiments

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt

## Run all experiments (batch mode)

To run all experiments across all `.vrp` instances in `data/`:

```bash
python -m src.run_experiments --instances_glob "data/*.vrp" --seeds 0:29 --n_iterations 30 --n_ants 10 --out_dir results_test
```

## Analyse the results
To analyse the results and generate the convergence plots and the statistical analysis:

```bash
python -m src.analyze_results \
  --results_dir results_test \
  --aggregate_ls_stats \
  --plot_all_instances \
  --stats \
  --score median \
  --alpha 0.05
```
## Smoke test with visualization

To run a quick smoke test with visualization on a single instance:

```bash
python -m src.main_smoketest --variant acs_cl15_ls2opt_fixed --problem-file data/X-n120-k6.vrp
```