<h1 align="center">ns3-ai-ntn / python_utils — RL extensions (W4)</h1>

<p align="center"><strong>Stable-Baselines3 + PyTorch Geometric + canonical ns3-gym layer for the <a href="https://github.com/Muhammaduazir69/ns3-ntn-toolkit">ns3-ntn-toolkit</a>.</strong></p>

<p align="center"><em>Part of the v2.0 roadmap (Workstream W4).</em></p>

---

## What it adds

The legacy `ns3ai_utils` shared-memory bridge is preserved; this directory adds a sibling Python package `ns3_ai_ntn` with:

- 4 Gymnasium environments wrapping `ntn-cho` / `ntn-rrc` / `ntn-observability`
- Stable-Baselines3 PPO/SAC training scripts
- PyTorch Geometric models for constellation-graph learning (GAT next-hop)
- Multi-Agent baselines (MAPPO / MASAC, parameter-shared)
- ns3-gym compatibility shim so legacy scripts work unmodified

```
ns3-ntn-toolkit
├── envs        (HandoverEnv, BeamMgmtEnv, SliceEnv, PowerCtrlEnv)
├── sb3         → PPO 10k-step trainer
├── gnn         → GAT on Starlink ISL graph
├── marl        → MAPPO + MASAC baselines
└── ns3gym_compat  → drop-in for `ns3gym.ns3env.Ns3Env`
```

## Install

```bash
cd contrib/ns3-ai-ntn/python_utils
pip install -e .[all]            # gymnasium + sb3 + pyg
# or, à la carte:
pip install -e .[sb3]
pip install -e .[gnn]
```

## Validation gates (all green)

| Gate | Result |
|---|---|
| `gymnasium.utils.env_checker.check_env` on all 4 envs | ✅ pass |
| `pytest contrib/ns3-ai-ntn/python_utils/tests/` | ✅ 15/15 |
| PPO beats random baseline by >1σ on `HandoverEnv` (10k steps) | ✅ gap 116, σ 27 |
| GAT next-hop accuracy on 50-node Starlink subset ≥ 70% | ✅ 88% |

## Quick start

### 1. Train PPO on the conditional-handover env

```bash
python -m ns3_ai_ntn.sb3.train_ppo_handover \
    --total-steps 10000 --eval-episodes 30 --seed 0 \
    --out /tmp/ppo_result.json
```

Reports mean ± std vs a random baseline and exit-codes 0 only when the trained policy clears the 1σ gate.

### 2. GNN forward pass on a Starlink subset

```python
import torch
from ns3_ai_ntn.gnn.constellation_graph import (
    build_pyg_data, starlink_subset_demo,
)
from ns3_ai_ntn.gnn.gat_topology import GATTopology, train_quick

demo = starlink_subset_demo(n_sats=50, seed=0)
data = build_pyg_data(demo["positions"], demo["edges"])
labels = torch.from_numpy(demo["labels"])
model, acc = train_quick(data, labels, epochs=300)
print(f"next-hop accuracy: {acc:.1%}")   # ~88%
```

The graph builder also accepts ECEF positions and ISL edges from the W1 `ntn-constellation` package, so you can plug a real Walker shell straight in:

```python
from ntn_constellation import propagator, isl
sats = propagator.from_preset("starlink-v1-shell1", limit=50)
positions = [s.eci_position(t0) for s in sats]
edges = isl.build_isl_topology(positions, k_nearest=4)
data = build_pyg_data(positions, edges)
```

### 3. ns3-gym compat shim

```python
from ns3_ai_ntn.ns3gym_compat import Ns3Env
env = Ns3Env(envName="HandoverEnv")
obs = env.reset()
obs, reward, done, info = env.step(env.action_space.sample())
```

The shim returns the legacy 4-tuple `(obs, reward, done, info)` so older scripts written against `ns3gym.ns3env.Ns3Env` work without modification. Pass `gymnasium_api=True` to opt in to the Gymnasium 1.0 5-tuple.

### 4. MARL baselines

```bash
python -m ns3_ai_ntn.marl.mappo_handover --n-agents 4 --total-steps 8000
python -m ns3_ai_ntn.marl.masac_beam --n-agents 2 --total-steps 4000
```

Both use parameter-shared policies; replace `make_vec_env` with one trainer per agent for fully independent IPPO/IDDPG.

## Tests

```bash
pip install -e .[test]
pytest tests/ -v
```

15 tests cover env contract conformance (check_env), reset determinism, action clipping, GAT forward shape, GAT accuracy gate, PPO 1k-step learn, ns3-gym compat smoke.

## Audit results (2026-05-04)

Stress-tested at 50 k PPO steps × 3 seeds and 80-sat GAT × 5 seeds × 1 000
epochs as part of the W1–W4 integration audit:

**PPO @ 50 000 steps × 3 seeds (vs random baseline, 50 eval episodes):**

| Seed | PPO mean ± σ | Random mean ± σ | Gap | Threshold (1σ) | σ-multiple |
|---:|---:|---:|---:|---:|---:|
| 0 | 137.99 ± 46.86 | 12.45 ± 25.74 | 125.5 | 25.7 | **~5σ** |
| 7 | 156.76 ± 28.58 | 12.99 ± 20.29 | 143.8 | 20.3 | **~7σ** |
| 42 | 140.01 ± 41.42 | 15.67 ± 24.53 | 124.3 | 24.5 | **~5σ** |

**GAT @ 80 sats × 1 000 epochs × 5 seeds:**

| seed | accuracy |
|:---:|---:|
| 0 | 0.950 |
| 1 | 0.962 |
| 2 | 0.938 |
| 3 | 0.988 |
| 4 | 0.925 |

Mean **95.2 %** — well above the 70 % gate, even at 1.6× the unit-test scale.

**W1→W4 integration:** GAT trained on 60-sat **real Walker-Star Starlink shell-1
geometry** (W1 propagator output, not the synthetic demo) reaches **90 %**
next-hop accuracy. The W1 `state_vectors()` API slots into
`build_pyg_data()` without coercion — confirming the cross-workstream
contract.

## License

GPL-2.0-only. Same as the parent `ns3-ntn-toolkit`.

## Maintainer

Muhammad Uzair — `muhammaduzairr69@gmail.com` (ORCID: 0009-0002-4104-2680)
