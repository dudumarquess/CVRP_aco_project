## Statistical Analysis (Demsar-style)

Instances used: 13
Variants: acs_baseline, acs_cl15, acs_cl15_ls2opt_adaptive, acs_cl15_ls2opt_adaptive_q0rho, acs_cl15_ls2opt_adaptive_q0sched, acs_cl15_ls2opt_fixed

### Average Ranks (lower is better)
| variant | avg_rank |
|---|---|
| acs_cl15_ls2opt_adaptive_q0sched | 1.654 |
| acs_cl15_ls2opt_adaptive_q0rho | 1.769 |
| acs_cl15_ls2opt_adaptive | 3.077 |
| acs_cl15_ls2opt_fixed | 3.500 |
| acs_baseline | 5.269 |
| acs_cl15 | 5.731 |

Friedman chi^2 = 55.7865, p = 8.992e-11

### Post-hoc (Holm vs control, alpha=0.05)
Control: **acs_cl15_ls2opt_adaptive_q0sched**
| variant | p | p_holm | significant |
|---|---|---|---|
| acs_baseline | 0.0002441 | 0.001221 | yes |
| acs_cl15 | 0.0002441 | 0.001221 | yes |
| acs_cl15_ls2opt_fixed | 0.002441 | 0.007324 | yes |
| acs_cl15_ls2opt_adaptive | 0.003418 | 0.007324 | yes |
| acs_cl15_ls2opt_adaptive_q0rho | 0.2036 | 0.2036 | no |