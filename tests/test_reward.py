from rl_token_minimize.reward import EpisodeOutcome, RewardConfig, compute_reward, patch_similarity

CFG = RewardConfig()


def outcome(**overrides):
    base = dict(
        protocol_error=False,
        finished=True,
        edited=True,
        tests_passed=True,
        touched_expected=True,
        similarity=1.0,
        unrelated_edits=False,
        tokens=500,
        tool_calls=4,
        failed_replaces=0,
    )
    base.update(overrides)
    return EpisodeOutcome(**base)


def test_protocol_error_is_strong_negative():
    assert compute_reward(outcome(protocol_error=True, tests_passed=False), CFG) == CFG.protocol_penalty


def test_no_edit_is_negative():
    assert compute_reward(outcome(edited=False, tests_passed=False), CFG) == CFG.no_edit_penalty


def test_failed_tests_with_similar_patch_gets_small_reward():
    r = compute_reward(outcome(tests_passed=False, similarity=0.9), CFG)
    assert r == CFG.near_miss_reward


def test_failed_tests_with_dissimilar_patch_gets_nothing():
    assert compute_reward(outcome(tests_passed=False, similarity=0.1), CFG) == 0.0


def test_failed_tests_wrong_file_gets_nothing():
    r = compute_reward(outcome(tests_passed=False, touched_expected=False, similarity=0.9), CFG)
    assert r == 0.0


def test_passing_gets_large_reward():
    assert compute_reward(outcome(), CFG) >= CFG.pass_reward


def test_fewer_tokens_rewarded_among_passing():
    lean = compute_reward(outcome(tokens=200), CFG)
    verbose = compute_reward(outcome(tokens=1800), CFG)
    assert lean > verbose


def test_failed_replaces_penalized_among_passing():
    clean = compute_reward(outcome(), CFG)
    sloppy = compute_reward(outcome(failed_replaces=3), CFG)
    assert clean > sloppy


def test_extra_tool_calls_penalized_among_passing():
    lean = compute_reward(outcome(tool_calls=3), CFG)
    wasteful = compute_reward(outcome(tool_calls=12), CFG)
    assert lean > wasteful


def test_unrelated_edits_penalized_among_passing():
    focused = compute_reward(outcome(), CFG)
    scattered = compute_reward(outcome(unrelated_edits=True), CFG)
    assert focused > scattered


def test_worst_pass_beats_best_near_miss():
    worst_pass = compute_reward(
        outcome(tokens=100_000, tool_calls=50, failed_replaces=20, unrelated_edits=True), CFG
    )
    best_miss = compute_reward(outcome(tests_passed=False, similarity=1.0), CFG)
    assert worst_pass > best_miss


def test_passing_with_protocol_error_still_scores_at_least_pass_floor():
    r = compute_reward(outcome(protocol_error=True), CFG)
    assert r >= CFG.pass_floor
    assert r != CFG.protocol_penalty


def test_unfinished_pass_scores_lower_than_finished_pass():
    finished = compute_reward(outcome(finished=True), CFG)
    unfinished = compute_reward(outcome(finished=False), CFG)
    assert finished > unfinished


def test_min_passing_reward_beats_max_non_passing_reward():
    assert CFG.pass_floor > CFG.near_miss_reward


def test_patch_similarity_identical_is_one():
    files = {"a.py": "x = 1\ny = 2\n"}
    assert patch_similarity(files, files) == 1.0


def test_patch_similarity_orders_closeness():
    ref = {"a.py": "def f():\n    return 1\n"}
    close = {"a.py": "def f():\n    return 2\n"}
    far = {"a.py": "import os\nprint('hello')\n"}
    assert patch_similarity(close, ref) > patch_similarity(far, ref)
