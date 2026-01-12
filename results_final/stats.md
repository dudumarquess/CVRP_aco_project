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