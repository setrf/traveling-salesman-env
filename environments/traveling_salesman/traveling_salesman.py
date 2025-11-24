import itertools
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from datasets import Dataset
from verifiers import Environment, Parser, Rubric
from verifiers.types import ChatMessage, RolloutInput, SamplingArgs, State


@dataclass
class TSPInstance:
    coords: List[Tuple[float, float]]
    distance_matrix: List[List[float]]
    optimal_route: List[int]
    optimal_distance: float
    start_city: int = 0


def build_distance_matrix(coords: Sequence[Tuple[float, float]]) -> List[List[float]]:
    matrix: List[List[float]] = []
    for i, (x1, y1) in enumerate(coords):
        row: List[float] = []
        for j, (x2, y2) in enumerate(coords):
            if i == j:
                row.append(0.0)
            else:
                row.append(math.dist((x1, y1), (x2, y2)))
        matrix.append(row)
    return matrix


def brute_force_optimal(distance_matrix: Sequence[Sequence[float]], start: int = 0) -> Tuple[List[int], float]:
    n = len(distance_matrix)
    cities = [i for i in range(n) if i != start]
    best_route: List[int] = [start]
    best_distance = math.inf
    for perm in itertools.permutations(cities):
        route = [start, *perm, start]
        dist = route_distance(distance_matrix, route)
        if dist < best_distance:
            best_distance = dist
            best_route = list(route)
    return best_route, best_distance


def route_distance(distance_matrix: Sequence[Sequence[float]], route: Sequence[int]) -> float:
    return sum(distance_matrix[a][b] for a, b in zip(route, route[1:]))


def generate_tsp_instance(
    num_cities: int,
    rng: random.Random,
    start_city: int = 0,
) -> TSPInstance:
    # Sample coordinates in a unit square
    coords = [(rng.random(), rng.random()) for _ in range(num_cities)]
    distance_matrix = build_distance_matrix(coords)
    optimal_route, optimal_distance = brute_force_optimal(distance_matrix, start=start_city)
    return TSPInstance(
        coords=coords,
        distance_matrix=distance_matrix,
        optimal_route=optimal_route,
        optimal_distance=optimal_distance,
        start_city=start_city,
    )


def format_question(instance: TSPInstance) -> str:
    coord_lines = [f"{idx}: ({x:.3f}, {y:.3f})" for idx, (x, y) in enumerate(instance.coords)]
    coord_block = "\n".join(coord_lines)
    return (
        "You are solving a Traveling Salesman Problem (TSP).\n"
        f"Start and end at city {instance.start_city}. Visit every city exactly once.\n"
        "Cities and coordinates:\n"
        f"{coord_block}\n\n"
        "Return the route as space-separated city indices, starting and ending at the start city.\n"
        "Example format: 0 2 3 1 0\n"
        "Only answer with the route."
    )


def parse_route_from_text(text: str, num_cities: int, start_city: int) -> Tuple[List[int], Dict[str, Any]]:
    """Extract a route from a model completion."""
    # Prefer the best line containing integers; fall back to a sliding window over all numbers.
    # This is intentionally lenient to avoid format penalties while salvaging a usable route from verbose text.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    all_numbers = [int(x) for x in re.findall(r"-?[0-9]+", text)]

    def score_line(nums: List[int]) -> int:
        # higher is better: complete coverage, correct length, starts/ends at start
        if not nums:
            return -1
        score = 0
        if nums[0] == start_city:
            score += 2
        if nums[-1] == start_city:
            score += 2
        score += min(len(set(nums)), num_cities)
        if len(nums) == num_cities + 1:
            score += 2
        return score

    best_line_nums: List[int] = []
    best_score = -1
    for ln in lines:
        nums = [int(x) for x in re.findall(r"-?[0-9]+", ln)]
        if len(nums) < 2:
            continue
        s = score_line(nums)
        if s > best_score:
            best_score = s
            best_line_nums = nums

    candidate_nums = best_line_nums if best_score >= 0 else all_numbers
    details: Dict[str, Any] = {"raw_numbers": candidate_nums, "all_numbers": all_numbers}

    if not candidate_nums:
        details.update({"feasible": False, "reason": "no_numbers"})
        return [], details

    # Try to build a route from candidate numbers; if that fails, slide over all numbers.
    def try_build(nums: List[int]) -> Tuple[List[int], Dict[str, Any]]:
        if not nums:
            return [], {"feasible": False, "reason": "no_numbers"}

        route = list(nums)
        if route[0] != start_city:
            route = [start_city] + route
        if route[-1] != start_city:
            route = route + [start_city]

        dedup_route = [route[0]]
        for node in route[1:]:
            if node != dedup_route[-1]:
                dedup_route.append(node)
        route = dedup_route

        inner = route[1:-1]
        unique_inner = set(inner)
        if len(route) != num_cities + 1 or len(unique_inner) != num_cities - 1:
            return route, {"feasible": False, "reason": "invalid_coverage", "route": route}
        if any(node < 0 or node >= num_cities for node in inner):
            return route, {"feasible": False, "reason": "out_of_range", "route": route}
        return route, {"feasible": True, "route": route}

    route, info = try_build(candidate_nums)
    if info.get("feasible"):
        details.update(info)
        return route, details

    # Fallback: slide window over all numbers to find a plausible tour length n+1
    n_needed = num_cities + 1
    best_route: List[int] = []
    for i in range(0, max(0, len(all_numbers) - n_needed + 1)):
        window = all_numbers[i : i + n_needed]
        r, inf = try_build(window)
        if inf.get("feasible"):
            details.update(inf)
            return r, details
        # keep the first near-feasible window as best guess
        if not best_route:
            best_route, best_info = r, inf

    details.update(best_info if 'best_info' in locals() else {"feasible": False, "reason": "invalid_coverage", "route": route})
    return best_route, details


def _coerce_message_to_text(message: Any) -> str:
    """
    OpenAI 4o/5.* models may return message.content as a list of parts.
    This helper extracts any text fields and joins them.
    """
    content = getattr(message, "content", "") or ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            # part can be dict-like or object with .text/.value
            if isinstance(part, str):
                parts.append(part)
                continue
            if isinstance(part, dict):
                text_val = part.get("text") or part.get("value") or ""
            else:
                text_val = getattr(part, "text", None) or getattr(part, "value", None) or ""
            if text_val:
                parts.append(str(text_val))
        return "\n".join(parts)

    try:
        return str(content)
    except Exception:
        return ""


class TravelingSalesmanEnv(Environment):
    def __init__(self, env_id: str = "traveling-salesman", env_args: Dict[str, Any] | None = None):
        self.env_args = env_args or {}
        cfg = {
            "train_examples": int(self.env_args.get("train_examples", 48)),
            "eval_examples": int(self.env_args.get("eval_examples", 16)),
            # Harder defaults: push city count up so tours are longer/non-trivial
            "min_cities": int(self.env_args.get("min_cities", 10)),
            "max_cities": int(self.env_args.get("max_cities", 10)),
            "seed": int(self.env_args.get("seed", 13)),
        }
        rng = random.Random(cfg["seed"])

        train_rows = [self._make_row(rng, cfg) for _ in range(cfg["train_examples"])]
        eval_rows = [self._make_row(rng, cfg) for _ in range(cfg["eval_examples"])]

        system_prompt = (
            "You are a routing expert. Respond with exactly one line of space-separated city indices, "
            "starting and ending at city 0 (e.g., 0 2 3 1 0). Do not include any other text, punctuation, "
            "or lines. Do NOT call tools or functions; return plain text only."
        )
        parser = Parser()
        rubric = Rubric(funcs=[self.score_route], parser=parser)

        super().__init__(
            dataset=Dataset.from_list(train_rows),
            eval_dataset=Dataset.from_list(eval_rows),
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            env_id=env_id,
            env_args=self.env_args,
            message_type="chat",
        )

    def _make_row(self, rng: random.Random, cfg: Dict[str, int]) -> Dict[str, Any]:
        num_cities = rng.randint(cfg["min_cities"], cfg["max_cities"])
        instance = generate_tsp_instance(num_cities=num_cities, rng=rng)
        question = format_question(instance)
        answer = " ".join(str(x) for x in instance.optimal_route)
        return {
            "question": question,
            "answer": answer,
            "info": {
                "distance_matrix": instance.distance_matrix,
                "optimal_distance": instance.optimal_distance,
                "optimal_route": instance.optimal_route,
                "num_cities": num_cities,
                "start_city": instance.start_city,
            },
        }

    async def setup_state(self, state: State) -> State:
        return state

    async def rollout(
        self,
        input: RolloutInput,
        client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        # Ensure we always ask for compact text responses to discourage chatter.
        merged_sampling: Dict[str, Any] = dict(sampling_args or {})
        merged_sampling.setdefault("max_tokens", 128)
        merged_sampling.setdefault("temperature", 0)

        # Some newer OpenAI models (gpt-5.*) return list-based content and can be touchy
        # about forced response_format. Relax that but keep using max_tokens to stay
        # compatible with inference gateways that don't accept max_output_tokens.
        model_lower = model.lower()
        if model_lower.startswith("openai/gpt-5") or model_lower.startswith("gpt-5"):
            merged_sampling.pop("response_format", None)
            merged_sampling.pop("max_output_tokens", None)
        else:
            merged_sampling.setdefault("response_format", {"type": "text"})

        state = await self.init_state(input, client, model, merged_sampling)
        prompt = state["prompt"]
        response = await self.get_model_response(
            client=client,
            model=model,
            prompt=prompt,
            oai_tools=None,
            sampling_args=merged_sampling,
            message_type=self.message_type,
        )

        completion_text: str
        if self.message_type == "chat":
            completion_text = _coerce_message_to_text(response.choices[0].message)
            assistant_msg: ChatMessage = {"role": "assistant", "content": completion_text}
            state["trajectory"].append(
                {"prompt": prompt, "completion": [assistant_msg], "reward": None, "advantage": None}
            )
        else:
            completion_text = response.choices[0].text or ""
            state["trajectory"].append(
                {"prompt": prompt, "completion": completion_text, "reward": None, "advantage": None}
            )

        state["completion"] = completion_text
        state["answer"] = input.get("answer", "")
        state["info"] = input.get("info", {})
        # Diagnostics for analysis/leaderboards
        state["info"]["completion_len_chars"] = len(completion_text)
        state["info"]["completion_lines"] = len([ln for ln in completion_text.splitlines() if ln.strip()])
        state["stop_condition"] = None
        return state

    async def score_route(self, completion: str, state: State, **_) -> float:
        info = state.get("info", {})
        num_cities = int(info.get("num_cities", 0))
        start_city = int(info.get("start_city", 0))
        distance_matrix = info.get("distance_matrix", [])
        optimal_distance = float(info.get("optimal_distance", 1e-9))

        route, details = parse_route_from_text(completion, num_cities, start_city)
        feasible = details.get("feasible", False)
        if not feasible or not distance_matrix or len(route) < 2:
            state["metrics"] = {"tsp_reward": 0.0, "feasible": 0.0, "reason": details.get("reason", "invalid")}
            state["reward"] = 0.0
            state["info"]["route_details"] = details
            return 0.0

        length = route_distance(distance_matrix, route)
        # reward: normalized inverse length (1.0 = optimal, approaches 0 as it gets worse)
        reward = max(0.0, min(1.0, optimal_distance / length))
        gap = length - optimal_distance

        state["reward"] = reward
        state["metrics"] = {
            "tsp_reward": reward,
            "tour_length": length,
            "optimal_length": optimal_distance,
            "gap": gap,
            "feasible": 1.0,
        }
        state["info"]["parsed_route"] = route
        state["info"]["route_details"] = details
        return reward


def load_environment(**kwargs) -> Environment:
    """
    Entrypoint for verifiers to load the environment.
    """
    return TravelingSalesmanEnv(env_args=kwargs)
