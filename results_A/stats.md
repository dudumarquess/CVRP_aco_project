## Statistical Analysis (Demsar-style)

Instances used: 15
Variants: acs_baseline, acs_cl15, acs_cl15_ls2opt_fixed, acs_cl15_ls2opt_fixed_q0sched, acs_clsqrtn_2opt_q0sched, acs_nocl_2opt_q0sched, acs_nocl_none

### Average Ranks (lower is better)
| variant | avg_rank |
|---|---|
| acs_cl15_ls2opt_fixed_q0sched | 2.200 |
| acs_clsqrtn_2opt_q0sched | 2.333 |
| acs_nocl_2opt_q0sched | 2.467 |
| acs_cl15_ls2opt_fixed | 3.000 |
| acs_baseline | 5.967 |
| acs_nocl_none | 5.967 |
| acs_cl15 | 6.067 |

Friedman chi^2 = 70.2972, p = 3.553e-13

### Post-hoc (Holm vs control, alpha=0.05)
Control: **acs_cl15_ls2opt_fixed_q0sched**
| variant | p | p_holm | significant |
|---|---|---|---|
| acs_baseline | 6.104e-05 | 0.0003662 | yes |
| acs_cl15 | 6.104e-05 | 0.0003662 | yes |
| acs_nocl_none | 6.104e-05 | 0.0003662 | yes |
| acs_nocl_2opt_q0sched | 0.2293 | 0.6879 | no |
| acs_cl15_ls2opt_fixed | 0.6378 | 1 | no |
| acs_clsqrtn_2opt_q0sched | 0.8261 | 1 | no |