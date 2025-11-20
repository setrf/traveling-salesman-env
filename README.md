# Traveling Salesman Prime-RL Environment

Turnkey verifiers/Prime-RL environment for the Traveling Salesman Problem (TSP). It ships a synthetic dataset of small TSP instances (4–7 cities) with ground-truth optimal tours and a rubric that scores model output as a normalized tour-length ratio.

## Repository layout
- `environments/traveling_salesman/pyproject.toml` — environment metadata (name, version, tags, deps)
- `environments/traveling_salesman/traveling_salesman.py` — environment implementation (dataset synth, prompt, rollout, scoring)
- `environments/traveling_salesman/README.md` — environment-level docs (args, metrics, quickstart)
- `.gitignore` — Python/build ignores

## Local setup
```bash
cd environments/traveling_salesman
uv pip install -e .      # or: pip install -e .
```

Quick sanity check (prints reward=1.0 for the optimal tour):
```bash
cd environments/traveling_salesman
python3 - <<'PY'
from traveling_salesman import load_environment
import asyncio
env = load_environment()
row = env.dataset[0]
state = {'prompt': row['prompt'], 'info': row['info'], 'completion': row['answer']}
reward = asyncio.run(env.score_route(completion=row['answer'], state=state))
print("reward", reward, "metrics", state.get("metrics"))
PY
```

## Using with verifiers
Evaluate with defaults:
```bash
uv run vf-eval traveling-salesman
```

Customize model/sampling and env args:
```bash
uv run vf-eval traveling-salesman \
  -m gpt-4o-mini \
  -n 20 -r 3 \
  -a '{"train_examples":64,"eval_examples":32,"min_cities":4,"max_cities":7,"seed":42}'
```

## Publish to Prime Intellect Environments Hub
From `environments/traveling_salesman/`:
```bash
prime env push --visibility PUBLIC
```
  - Uses `pyproject.toml` name/version (currently 0.1.4).
- Add `--team <team-slug>` if you want team ownership.

After push:
- Hub page: `https://app.primeintellect.ai/dashboard/environments/<owner>/traveling-salesman`
- Install: `prime env install <owner>/traveling-salesman` or `uv pip install traveling-salesman --extra-index-url https://hub.primeintellect.ai/<owner>/simple/`

## Training notes (PRIME-RL)
1) Spin up a pod with your credits.
2) Run inference + orchestrator + trainer pointed at `traveling-salesman`.
3) Start with a small model (e.g., 7B–8B) and short rollouts; reward is `optimal_distance / tour_distance` (1.0 optimal, 0 for invalid, -1 for infeasible parsing).
4) Override env args for more/less cities or larger eval sets. Longer tours increase reasoning difficulty.

## GitHub publishing
After the first commit (already staged below), push to a new public repo:
```bash
git remote add origin git@github.com:<you>/traveling-salesman-env.git
git push -u origin main
```
