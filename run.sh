#!/usr/bin/env bash
set -euo pipefail

# =========================
# FINAL RUN CONFIG
# =========================
OUT_DIR="results_final"
SEEDS="0:29"          # 30 seeds
N_ITERS=30
N_ANTS=10

# Instâncias (ajuste se seu dataset estiver em outro lugar)
DATA_GLOB="data/*.vrp"

# Variantes finais (removi duplicata baseline vs nocl_none: escolha UMA)
VARIANTS=(
  "acs_baseline"
  "acs_cl15"
  "acs_cl15_ls2opt_fixed"
  "acs_cl15_ls2opt_fixed_q0sched"
  "acs_clsqrtn_2opt_q0sched"
  "acs_nocl_2opt_q0sched"
)

echo "== Cleaning output dir: ${OUT_DIR} =="
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

echo "== Git commit =="
git rev-parse --short HEAD || true

echo "== Running all instances =="
python -m src.run_experiments \
  --instances_glob "${DATA_GLOB}" \
  --variants "${VARIANTS[@]}" \
  --seeds "${SEEDS}" \
  --n_iterations "${N_ITERS}" \
  --n_ants "${N_ANTS}" \
  --out_dir "${OUT_DIR}"


echo "== Analyzing + stats (summary + long csv + demšar stats + cd plot) =="
python -m src.analyze_results \
  --results_dir "${OUT_DIR}" \
  --aggregate_ls_stats \
  --stats \
  --score median \
  --alpha 0.05

echo "== DONE. Outputs in ${OUT_DIR}/ =="
echo " - ${OUT_DIR}/summary.csv"
echo " - ${OUT_DIR}/summary.md"
echo " - ${OUT_DIR}/runs_long.csv"
echo " - ${OUT_DIR}/stats.md"
echo " - ${OUT_DIR}/stats_cd.png"
