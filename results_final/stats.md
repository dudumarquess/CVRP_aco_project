## Statistical Analysis (Demsar-style)

Instances used: 15
Variants: acs_baseline, acs_cl15, acs_cl15_ls2opt_fixed, acs_cl15_ls2opt_fixed_q0sched, acs_clsqrtn_2opt_q0sched, acs_nocl_2opt_q0sched

### Average Ranks (lower is better)
| variant | avg_rank |
|---|---|
| acs_clsqrtn_2opt_q0sched | 2.133 |
| acs_nocl_2opt_q0sched | 2.267 |
| acs_cl15_ls2opt_fixed_q0sched | 2.533 |
| acs_cl15_ls2opt_fixed | 3.067 |
| acs_baseline | 5.267 |
| acs_cl15 | 5.733 |

Friedman chi^2 = 54.5010, p = 1.653e-10

### Post-hoc (Holm vs control, alpha=0.05)
Control: **acs_clsqrtn_2opt_q0sched**
| variant | p | p_holm | significant |
|---|---|---|---|
| acs_baseline | 6.104e-05 | 0.0003052 | yes |
| acs_cl15 | 6.104e-05 | 0.0003052 | yes |
| acs_cl15_ls2opt_fixed | 0.3305 | 0.9916 | no |
| acs_nocl_2opt_q0sched | 0.5995 | 1 | no |
| acs_cl15_ls2opt_fixed_q0sched | 0.6832 | 1 | no |

### Aggregate Summary (Table 1)
| Variant | Rank | Holm | Time A (s) | Time X (s) | CostA | CostX |
|---|---:|:---:|---:|---:|---:|---:|
| acs_clsqrtn_2opt_q0sched | 2.133 | – | 0.108 | 0.605 | 982.57 | 16754.96 |
| acs_nocl_2opt_q0sched | 2.267 | no | 0.142 | 0.965 | 976.34 | 16883.48 |
| acs_cl15_ls2opt_fixed_q0sched | 2.533 | no | 0.117 | 0.621 | 986.30 | 16698.89 |
| acs_cl15_ls2opt_fixed | 3.067 | no | 0.112 | 0.594 | 995.26 | 16689.28 |
| acs_baseline | 5.267 | yes | 0.126 | 0.798 | 1031.00 | 17428.34 |
| acs_cl15 | 5.733 | yes | 0.099 | 0.463 | 1037.90 | 17308.87 |