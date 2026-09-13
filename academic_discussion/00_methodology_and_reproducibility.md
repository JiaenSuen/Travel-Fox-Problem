# Methodology and Reproducibility

## 1. Benchmark position

TFP is a compact partially observable reinforcement-learning benchmark inspired by mobile-robot logistics rather than a simulator of vehicle physics. Its research value comes from controlled task structure: the same PPO/evaluation stack is used across environments whose dominant difficulty changes from short behavioral cycles to multi-goal state, room topology, moving objects, access dependencies, and dynamic hazards. This keeps ablations computationally accessible while making failure mechanisms interpretable.

The design follows the same minimal-environment motivation that makes MiniGrid useful for goal-oriented RL research: a small state/action interface permits rapid task construction and controlled experimentation. TFP extends that philosophy toward transport and interaction problems with explicit deployment diagnostics, task-scoped rewards, and safety events.

## 2. Formal view

Each task can be treated as a POMDP with hidden global state \(s_t\), local observation \(o_t\), discrete action \(a_t\), task reward \(r_t\), and potentially history-dependent optimal behavior. The policy is trained with PPO. The observation deliberately omits the global map in normal operation; larger tasks may expose bounded task cues such as a next-doorway vector, inventory state, or observed hazard motion. These cues are intended to reduce irrelevant search complexity without exposing the full solution sequence.

The central experimental question is therefore not whether an oracle planner can solve the grid. It is whether a compact policy can transform limited observations and low-bandwidth temporal context into reliable interaction behavior.

## 3. Training protocol

The PPO implementation uses a frozen configuration object. Once a run begins, task, model, reward, action-mask mode, view size, learning rate, seed, rollout size, number of environments, and other hyperparameters are represented by an immutable snapshot. A SHA-256-derived protocol fingerprint is stored with checkpoints to make accidental protocol drift visible.

TFP Studio locks the corresponding controls while a worker is active. Pause/Resume affects only execution scheduling: rollout collection blocks on a condition variable and resumes with the same model, optimizer, environments, and configuration. Graceful Stop exits at a safe boundary and writes a distinct `.interrupted.pt` checkpoint. This is preferred to killing the training process because partially written checkpoints and ambiguous final updates are poor experimental artifacts.

## 4. Evaluation protocol

Controlled evaluation uses the packaged held-out map set crossed with fixed default seeds. Episode-level records retain success, completion, pickup, return, steps, path efficiency, collisions, invalid actions, cycle diagnostics, revisit rate, and inference latency. Dynamic-safety tasks additionally export hazard collision count, near-miss count, and failure reason. These disaggregated signals are important because two policies with the same success rate can fail for fundamentally different reasons.

Results are separated by task namespace. Formal cross-model conclusions should use the complete controlled matrix; development runs are useful for debugging but are labeled separately in the academic discussion.

## 5. Reward methodology

A repeated historical lesson in TFP is that a dense reward must correspond to **executable progress**, not merely geometric proximity. If forward motion earns more than reverse motion loses, an oscillating policy can create positive return without completing the task. Likewise, paying a fixed positive reward for opening a door can create a toggle loop unless closing and reopening are handled consistently.

For this reason, later TFP rewards use symmetric potential-like differences in geodesic or action-cost distance wherever possible. Door traversal can include the cost of `TOGGLE + MOVE`; milestone bonuses are reserved for irreversible or task-relevant transitions such as acquiring a key, picking up cargo, or completing delivery. These choices are motivated by the policy-invariance perspective of potential-based reward shaping, while acknowledging that practical TFP rewards also contain non-potential penalties and milestones and therefore require empirical ablation.

## 6. Action masking and deployment controllers

TFP distinguishes learning-time action masking from deployment-time anti-deadlock control. Invalid-action masking can improve discrete policy optimization when many actions are impossible, but it changes the exploration process and must remain fixed within a controlled run. Tabu and TabuX instead act after neural inference to suppress recognized short cycles or repeated local basins. The distinction allows an experiment to ask whether competence was learned by the policy or recovered by a lightweight runtime controller.

## 7. Safety interpretation

A soft safety prior or reward penalty is not a formal guarantee. In TFP, a "shield" in model names refers to a bounded action-logit safety bias derived from local hazard observations, not a formally synthesized temporal-logic shield. This terminology is intentionally clarified because formal safe-RL shielding has a stronger meaning in the literature. The TFP mechanism should therefore be evaluated empirically through collision rate, near misses, failure causes, and task completion rather than described as certified safety.

## 8. Threats to validity

TFP abstracts away continuous dynamics, sensing noise, localization error, actuator delay, and real collision geometry. Grid distance can make reward shaping unusually informative. Procedural seeds are deterministic and may not capture real-world nonstationarity. Low-bandwidth route cues encode task knowledge that a fully end-to-end robot would need to estimate. Finally, a small number of controlled runs cannot establish statistical significance across training seeds.

The intended use is therefore **mechanism study and prototype ablation**, not proof that a method will transfer directly to a physical robot. Strong conclusions should be followed by multi-seed training and, where appropriate, a higher-fidelity simulator or hardware experiment.
