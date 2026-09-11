# FOX-CS-L2 — Fox Color Sort · Local

Controlled cross-model benchmark generated from complete evaluation records.

| Model | Params | Runs | N | Success ↑ | Completion ↑ | Pickup ↑ | Steps ↓ | Return ↑ | Path Eff. ↑ | Collision ↓ | Invalid ↓ | Cycles ↓ | Interact cycles ↓ | Revisit ↓ | Infer ms ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `001_color_cnn` | 125.3k | 1 | 75 | 84.0% | 90.4% | 100.0% | 82.7 | 12.05 | 0.969 | 0.00 | 0.00 | 53.67 | 0.00 | 14.7% | 0.624 |
| `002_goal_conditioned_cnn` | 233.8k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `003_goal_gru_action_memory` | 144.9k | 1 | 75 | 89.3% | 92.2% | 98.7% | 65.2 | 12.84 | 0.975 | 0.00 | 0.00 | 35.31 | 0.00 | 10.0% | 1.832 |

*Controlled protocol: 15 test maps × 5 seeds = 75 episodes/run, `local` 5×5, `001_dense_color_sort`, `task` mask, `001_ppo_categorical`. Only complete protocol-matched runs enter this table; repeated runs are averaged. `—` means no qualifying benchmark yet.*

![Controlled benchmark summary](summary.png)

Ad-hoc, smoke, and ablation evaluations remain visible in **TFP Studio → Compare** but do not alter this table.
