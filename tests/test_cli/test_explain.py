"""`coastline explain` — the score-breakdown verb.

Pins Kavier + `--feasibility rules` like the other CLI tests. The assertions check that what is
rendered agrees with what the policy actually computed (the weighted sum, the ordering, the
min_gpu special case), never a specific engine number.
"""

from __future__ import annotations

import re

import pytest

from coastline.cli import main

_BASE = [
    "explain",
    "--model",
    "mistral-7b-v0.1",
    "--method",
    "lora",
    "--gpu-model",
    "NVIDIA-A100-SXM4-80GB",
    "--tokens",
    "1024",
    "--batch-size",
    "16",
    "--predictor",
    "kavier",
    "--feasibility",
    "rules",
]


def _rows(out: str) -> list[list[str]]:
    """The candidate table rows, as lists of cells."""
    return [line.split() for line in out.splitlines() if re.match(r"^\s+\d+\s+\d+x\d+\s", line)]


def test_explain_renders_the_ranked_candidates_with_score_components(capsys) -> None:
    main([*_BASE, "--top-k", "4"])
    out = capsys.readouterr().out

    assert "rank  gpus  batch" in out
    for column in ("p_score", "t_score", "combined"):
        assert column in out

    rows = _rows(out)
    assert 1 <= len(rows) <= 4
    # Ranks are dense and ascending from 1.
    assert [int(r[0]) for r in rows] == list(range(1, len(rows) + 1))
    # combined_score is the ranking key, so it must be non-increasing down the table.
    combined = [float(r[-1]) for r in rows]
    assert combined == sorted(combined, reverse=True)


def test_explain_shows_the_weighted_sum_the_policy_evaluated(capsys) -> None:
    main([*_BASE, "--preset", "balanced", "--top-k", "1"])
    out = capsys.readouterr().out

    assert "alpha=0.50 power, beta=0.50 time" in out
    # The printed arithmetic must actually add up: alpha*power + beta*time == combined.
    match = re.search(
        r"score\s+([\d.]+) x ([\d.]+) \(power\) \+ ([\d.]+) x ([\d.]+) \(time\) = ([\d.]+)",
        out,
    )
    assert match is not None, out
    alpha, power_score, beta, throughput_score, combined = (float(g) for g in match.groups())
    assert alpha * power_score + beta * throughput_score == pytest.approx(combined, abs=5e-3)


def test_explain_presets_shift_the_weights(capsys) -> None:
    """alpha weighs power and beta weighs time (selection.py), so energy is alpha-heavy."""
    main([*_BASE, "--preset", "energy", "--top-k", "1"])
    energy_out = capsys.readouterr().out
    main([*_BASE, "--preset", "performance", "--top-k", "1"])
    performance_out = capsys.readouterr().out

    assert "alpha=0.80 power, beta=0.20 time" in energy_out
    assert "alpha=0.20 power, beta=0.80 time" in performance_out


def test_explain_min_gpu_hides_the_score_it_does_not_have(capsys) -> None:
    """min_gpu ranks on GPU count; its combined_score is a 1/total_gpus ordering proxy."""
    main([*_BASE, "--strategy", "min_gpu", "--top-k", "3"])
    out = capsys.readouterr().out

    assert "no weighted score" in out
    assert "combined" not in out
    assert "alpha=" not in out
    # It still reports the winner and the raw predictions.
    assert "winner" in out
    assert _rows(out)


def test_explain_min_gpu_picks_no_more_gpus_than_balanced(capsys) -> None:
    main([*_BASE, "--strategy", "min_gpu", "--top-k", "1"])
    min_gpu_out = capsys.readouterr().out
    main([*_BASE, "--strategy", "multi_objective", "--preset", "balanced", "--top-k", "1"])
    balanced_out = capsys.readouterr().out

    def winner_gpus(out: str) -> int:
        match = re.search(r"winner\s+(\d+) GPU", out)
        assert match is not None, out
        return int(match.group(1))

    assert winner_gpus(min_gpu_out) <= winner_gpus(balanced_out)


def test_explain_states_why_the_winner_won(capsys) -> None:
    main([*_BASE, "--top-k", "3"])
    out = capsys.readouterr().out

    assert "winner" in out
    assert "why" in out
    assert "feas" in out


def test_explain_rejects_a_misspelled_preset(capsys) -> None:
    """Unvalidated, a typo falls through the strategy's unconditional balanced fallback and
    explains a policy the user did not ask for."""
    with pytest.raises(SystemExit) as excinfo:
        main([*_BASE, "--preset", "perfomance"])

    assert excinfo.value.code == 2


def test_explain_discloses_the_throughput_tie_break(capsys) -> None:
    """selection.py reorders the top 0.01 band by throughput, so `combined` can legitimately read
    non-monotonically. Unexplained, that looks like a ranking bug."""
    main([*_BASE, "--preset", "performance", "--top-k", "5"])
    out = capsys.readouterr().out

    combined = [float(r[-1]) for r in _rows(out)]
    if combined != sorted(combined, reverse=True):
        assert "tie-break" in out
