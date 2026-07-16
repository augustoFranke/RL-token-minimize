import difflib
from dataclasses import dataclass


@dataclass
class EpisodeOutcome:
    protocol_error: bool
    finished: bool
    edited: bool
    tests_passed: bool
    touched_expected: bool
    similarity: float
    unrelated_edits: bool
    tokens: int
    tool_calls: int
    failed_replaces: int


@dataclass
class RewardConfig:
    protocol_penalty: float = -1.0
    no_edit_penalty: float = -0.5
    near_miss_reward: float = 0.2
    near_miss_similarity: float = 0.5
    pass_reward: float = 1.0
    pass_floor: float = 0.5
    token_bonus_weight: float = 0.5
    token_budget: int = 2048
    free_tool_calls: int = 4
    tool_call_penalty: float = 0.02
    failed_replace_penalty: float = 0.05
    unrelated_edit_penalty: float = 0.2
    unfinished_penalty: float = 0.1


def compute_reward(o: EpisodeOutcome, cfg: RewardConfig) -> float:
    if o.tests_passed:
        reward = cfg.pass_reward
        reward += cfg.token_bonus_weight * max(0.0, 1.0 - o.tokens / cfg.token_budget)
        reward -= cfg.tool_call_penalty * max(0, o.tool_calls - cfg.free_tool_calls)
        reward -= cfg.failed_replace_penalty * o.failed_replaces
        if o.unrelated_edits:
            reward -= cfg.unrelated_edit_penalty
        if not o.finished:
            reward -= cfg.unfinished_penalty
        return max(cfg.pass_floor, reward)
    if o.protocol_error:
        return cfg.protocol_penalty
    if not o.edited:
        return cfg.no_edit_penalty
    if o.touched_expected and o.similarity >= cfg.near_miss_similarity:
        return cfg.near_miss_reward
    return 0.0


def patch_similarity(final_files: dict[str, str], reference_files: dict[str, str]) -> float:
    paths = sorted(set(final_files) | set(reference_files))
    ratios = [
        difflib.SequenceMatcher(
            a=final_files.get(p, ""), b=reference_files.get(p, ""), autojunk=False
        ).ratio()
        for p in paths
    ]
    return sum(ratios) / len(ratios) if ratios else 1.0
