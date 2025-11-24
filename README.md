# Traveling Salesman Prime-RL Environment

Turnkey verifiers/Prime-RL environment for the Traveling Salesman Problem (TSP). It ships a synthetic dataset of medium TSP instances (default 10 cities) with ground-truth optimal tours and a rubric that scores model output as a normalized tour-length ratio.

## Repository layout
- `environments/traveling_salesman/pyproject.toml` — environment metadata (name, version, tags, deps)
- `environments/traveling_salesman/traveling_salesman.py` — environment implementation (dataset synth, prompt, rollout, scoring)
- `environments/traveling_salesman/README.md` — environment-level docs (args, metrics, quickstart)
- `.gitignore` — Python/build ignores (we avoid committing eval outputs)
- `outputs/` — not tracked; local eval artifacts only (do not commit)

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
  -a '{"train_examples":64,"eval_examples":32,"min_cities":10,"max_cities":10,"seed":42}'
```

Strict output contract:
- System prompt enforces: single line of space-separated city indices, start/end at 0, no extra text or tools.
- Rollout defaults: `temperature=0`, `max_tokens=128`.
- Parser picks the best numeric line (or sliding window) and scores invalid/empty outputs as 0 (not -1).

## Publish to Prime Intellect Environments Hub
From `environments/traveling_salesman/`:
```bash
prime env push --visibility PUBLIC
```
- Uses `pyproject.toml` name/version (currently 0.1.5).
- Add `--team <team-slug>` if you want team ownership.

After push:
- Hub page: `https://app.primeintellect.ai/dashboard/environments/<owner>/traveling-salesman`
- Install: `prime env install <owner>/traveling-salesman` or `uv pip install traveling-salesman --extra-index-url https://hub.primeintellect.ai/<owner>/simple/`

## Training notes (PRIME-RL)
1) Spin up a pod with your credits.
2) Run inference + orchestrator + trainer pointed at `traveling-salesman`.
3) Start with a small model (e.g., 7B–8B) and short rollouts; reward is `optimal_distance / tour_distance` (1.0 optimal, 0 for invalid, -1 for infeasible parsing).
4) Override env args for more/less cities or larger eval sets. Longer tours increase reasoning difficulty.

## Model notes (as of 0.2.1)
- Default difficulty uses 10-city graphs. Parser is lenient: it salvages the best numeric line (or sliding window) and scores invalids as 0.
- OpenAI gpt-5.1 SKUs remain unavailable on Prime Inference (404/not_found). Working models we’ve recently evaluated:  
  - gemini-3-pro-preview: 1.000 (4 ex)  
  - grok-4: 0.982 (4 ex)  
  - qwen3-coder: 0.900 (24 ex)  
  - kimi-k2-0905: 0.921 (3 ex)  
  - claude-sonnet-4.5: 0.977 (8 ex)  
  - llama-4-maverick: ~0.90 (earlier run)  
  - glm-4.6: new hard run pending; earlier small run completed.

## Recent evals (hard config, 10-city graphs)
- `google/gemini-3-pro-preview`: ID `j036cfk189dd7x2i7nhgm9rd` (4×1, avg 1.000)
- `x-ai/grok-4`: ID `tkhe95pulevisebiexea5ozs` (4×1, avg 0.982)
- `qwen/qwen3-coder`: ID `tqszrvj97gdx6h36cpn8d52i` (24×1, avg 0.900)
- `kimi-k2-0905`: ID `q6j4agm8zar1wd9oo86kolmp` (3×1, avg 0.921)
- `claude-sonnet-4.5`: ID `iwel728n21yuoxmqy1l9nka4` (8×1, avg 0.977)
- `glm-4.6`: new 6-sample hard run pending (earlier small run ID `vdzio6ha9e3h87vgwtiya4wz`).

## GitHub publishing
After the first commit (already staged below), push to a new public repo:
```bash
git remote add origin git@github.com:<you>/traveling-salesman-env.git
git push -u origin main
```
