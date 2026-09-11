# FOX-RM-L3 — Fox Room Transport · Doors

Controlled cross-model benchmark generated from complete evaluation records.

| Model | Params | Runs | N | Success ↑ | Completion ↑ | Pickup ↑ | Steps ↓ | Return ↑ | Path Eff. ↑ | Collision ↓ | Invalid ↓ | Cycles ↓ | Interact cycles ↓ | Revisit ↓ | Infer ms ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `001_route_prior_cnn` | 393.2k | 1 | 100 | 92.0% | 92.0% | 95.0% | 103.7 | 22.98 | 1.000 | 0.00 | 0.00 | 58.05 | 0.00 | 7.6% | 2.165 |
| `002_route_prior_action_memory` | 422.3k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `003_route_prior_gru_memory` | 520.8k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*Controlled protocol: 20 test maps × 5 seeds = 100 episodes/run, `local` 7×7, `001_actionable_geodesic`, `task` mask, `001_ppo_categorical`. Only complete protocol-matched runs enter this table; repeated runs are averaged. `—` means no qualifying benchmark yet.*

![Controlled benchmark summary](summary.png)

Ad-hoc, smoke, and ablation evaluations remain visible in **TFP Studio → Compare** but do not alter this table.
