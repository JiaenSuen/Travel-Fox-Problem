from __future__ import annotations

REWARD_SPEC = {
    "key": "001_sparse_transport",
    "display_name": "001 · Sparse Transport",
    "description": "Harder ablation: small step/invalid penalties with pickup and delivery rewards but no navigation progress shaping.",
}


class SparseTransportReward001:
    def compute(self, transition: dict[str, object]) -> float:
        reward = -0.01
        if bool(transition.get("invalid", False)):
            reward -= 0.08
        if bool(transition.get("pickup", False)):
            reward += 1.0
        if bool(transition.get("delivered", False)):
            reward += 8.0
        return float(reward)


def create_reward() -> SparseTransportReward001:
    return SparseTransportReward001()
