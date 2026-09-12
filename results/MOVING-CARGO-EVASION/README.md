# MOVING-CARGO-EVASION — Moving Cargo & Predator Avoidance

| Model | Params | Runs | N | Success ↑ | Completion ↑ | Pickup ↑ | Steps ↓ | Return ↑ | Path Eff. ↑ | Collision ↓ | Invalid ↓ | Cycles ↓ | Interact cycles ↓ | Revisit ↓ | Infer ms ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `001_intercept_safety_cnn` | 446.8k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `002_intercept_action_memory` | 478.7k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `003_intercept_gru_memory` | 615.3k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*Evaluation: 15 test maps × 5 seeds = 75 episodes/run · `local` 7×7 · `001_intercept_safety_potential` · `task` mask · `001_ppo_categorical`. `—` = pending.*

![Benchmark summary](summary.png)
