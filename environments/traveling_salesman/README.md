# traveling-salesman

Single-turn TSP routing environment for verifiers / Prime-RL. Each example is a medium TSP instance (default 10 cities) specified by coordinates. The agent must return a tour as space-separated city indices starting/ending at city 0. Reward is `optimal_distance / tour_distance` (1.0 = optimal, 0 for invalid).

### Overview
- **Environment ID**: `traveling-salesman`
- **Short description**: Generate TSP tours for randomly sampled small graphs.
- **Tags**: `tsp`, `rl`, `graphs`, `routing`, `eval`

### Data splits
- Train: 48 synthetic instances (by default)
- Eval: 16 synthetic instances (by default)
- Deterministic sampling via `seed` env arg; each row stores coords, distance matrix, optimal tour/length.

### Task
- **Type**: single-turn chat (completion also works)
- **Parser**: default verifiers `Parser`
- **Rubric**: parses the route, checks feasibility, computes tour length, reward = `optimal/tour` (clipped to [0,1]).

### Output format
- Return a tour as space-separated city indices, starting/ending at city 0 (e.g., `0 2 3 1 0`).
- No extra text or units; only the sequence (one line).
- The parser enforces:
  - Starts/ends at start city (0)
  - Visits every city exactly once
  - No out-of-range indices
  - No empty outputs

### Quickstart
Evaluate with defaults:
```bash
uv run vf-eval traveling-salesman
```

Change model/sampling and override env args:
```bash
uv run vf-eval traveling-salesman \
  -m gpt-4o-mini \
  -n 20 -r 3 \
  -a '{"train_examples": 64, "eval_examples": 32, "min_cities": 10, "max_cities": 10, "seed": 42}'
```

Sampling defaults baked into rollout:
- `temperature=0`
- `max_tokens=128`
- Parser picks the best numeric line (or a sliding numeric window) and salvages routes; invalid/empty outputs score 0 (not -1).
- You can add `-S '{"stop":["\\n",".",","]}'` to vf-eval to further trim verbosity if a model is chatty.

### Environment arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `train_examples` | int | `48` | Synthetic train rows to generate |
| `eval_examples` | int | `16` | Synthetic eval rows to generate |
| `min_cities` | int | `10` | Minimum cities per instance |
| `max_cities` | int | `10` | Maximum cities per instance |
| `seed` | int | `13` | RNG seed for reproducible instances |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `tsp_reward` | Main scalar reward (1.0 optimal, 0 invalid) |
| `tour_length` | Length of returned tour |
| `optimal_length` | Optimal tour length for the instance |
| `gap` | `tour_length - optimal_length` |
| `feasible` | 1 if the tour is valid/visits all cities once; else 0 |

### Notes on scoring
- Invalid or malformed routes get `tsp_reward = 0.0`.
- Feasible but suboptimal routes get a fractional reward: `optimal_distance / tour_distance` (clipped to [0,1]).
- Optimal route yields `tsp_reward = 1.0`.

### Outputs directory
An `outputs/` directory is packaged with the environment (contains a README and .gitkeep) to support automated evaluators that expect a writable outputs path during integration tests. You can also drop rollout artifacts there if needed.
