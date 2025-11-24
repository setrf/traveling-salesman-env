# Traveling Salesman Prime-RL Environment

Turnkey verifiers/Prime-RL environment for the Traveling Salesman Problem (TSP). It ships a synthetic dataset of small TSP instances (4–7 cities) with ground-truth optimal tours and a rubric that scores model output as a normalized tour-length ratio.

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
  -a '{"train_examples":64,"eval_examples":32,"min_cities":4,"max_cities":7,"seed":42}'
```

Strict output contract:
- System prompt enforces: single line of space-separated city indices, start/end at 0, no extra text.
- Rollout defaults: `response_format={"type": "text"}`, `temperature=0`, `max_tokens=128`.
- Parser extracts the first numeric line; invalid/empty outputs score -1.

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

## OpenAI model notes (as of 0.1.7)
- Env now unwraps list-based `message.content` and relaxes sampling args for `openai/gpt-5*` (drops forced `response_format=text`, adds `max_output_tokens` fallback), which resolves prior empty-content/404 issues seen on TSP prompts.
- If a provider still returns a 404 for `gpt-5.1*`, try rerunning with `-S '{"max_output_tokens":128}'` to mirror the baked defaults and confirm the SKU is enabled in your account.
- Other models (e.g., `qwen/qwen3-coder`, `google/gemini-3-pro-preview`, `anthropic/claude-sonnet-4.5`) respond; keep using them if OpenAI SKUs are rate-limited or unavailable.

## Recent evals (harder config: 6–9 cities, 96/48 splits, 48×3 rollouts)
- `qwen/qwen3-coder`: avg reward ~0.919. Results saved locally under `outputs/evals/setrf/traveling-salesman--qwen--qwen3-coder/79375a1a` (not tracked).

## GitHub publishing
After the first commit (already staged below), push to a new public repo:
```bash
git remote add origin git@github.com:<you>/traveling-salesman-env.git
git push -u origin main
```
