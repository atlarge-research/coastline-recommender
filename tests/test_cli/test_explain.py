"""Tests for `coastline explain`, which prints the score breakdown of the ranked candidates.

Runs pin Kavier and `--feasibility rules`, like the other CLI tests. The assertions check that the
printed table agrees with what the policy computed (the weighted sum, the ordering, the min_gpu
case) and do not depend on Kavier's numbers.
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

    # The runtime score (alpha's) comes before the energy score (beta's), as in the score line.
    assert "rank  gpus  batch   thr(tok/s)     P(W)  r_score  e_score  combined" in out

    rows = _rows(out)
    assert 1 <= len(rows) <= 4
    # Ranks are dense and ascending from 1.
    assert [int(r[0]) for r in rows] == list(range(1, len(rows) + 1))
    # combined_score is the ranking key, so it is non-increasing down the table, unless near-ties
    # were reordered by throughput, which the output then says.
    combined = [float(r[-1]) for r in rows]
    assert combined == sorted(combined, reverse=True) or "tie-break" in out


def test_explain_shows_the_weighted_sum_the_policy_evaluated(capsys) -> None:
    main([*_BASE, "--preset", "balanced", "--top-k", "1"])
    out = capsys.readouterr().out

    assert "alpha=0.50 runtime, beta=0.50 energy" in out
    # The printed terms add up: alpha*runtime + beta*energy == combined.
    match = re.search(
        r"score\s+([\d.]+) x ([\d.]+) \(runtime\) \+ ([\d.]+) x ([\d.]+) \(energy\) = ([\d.]+)",
        out,
    )
    assert match is not None, out
    alpha, throughput_score, beta, power_score, combined = (float(g) for g in match.groups())
    assert alpha * throughput_score + beta * power_score == pytest.approx(combined, abs=5e-3)
    # The table row shows the same two scores, runtime first.
    r_score, e_score = (float(cell) for cell in _rows(out)[0][-3:-1])
    assert (r_score, e_score) == pytest.approx((throughput_score, power_score), abs=5e-3)


def test_explain_presets_shift_the_weights(capsys) -> None:
    """alpha weighs runtime and beta weighs energy (selection.py), so energy is beta-heavy."""
    main([*_BASE, "--preset", "energy", "--top-k", "1"])
    energy_out = capsys.readouterr().out
    main([*_BASE, "--preset", "performance", "--top-k", "1"])
    performance_out = capsys.readouterr().out

    assert "alpha=0.20 runtime, beta=0.80 energy" in energy_out
    assert "alpha=0.80 runtime, beta=0.20 energy" in performance_out


def test_explain_min_gpu_hides_the_score_it_does_not_have(capsys) -> None:
    """min_gpu picks by feasibility; it has no scores, so the table shows only predictions."""
    main([*_BASE, "--strategy", "min_gpu", "--top-k", "3"])
    out = capsys.readouterr().out

    assert "no weighted score" in out
    assert "combined" not in out
    assert "r_score" not in out and "e_score" not in out
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
    """A misspelled preset is a usage error (exit 2)."""
    with pytest.raises(SystemExit) as excinfo:
        main([*_BASE, "--preset", "perfomance"])

    assert excinfo.value.code == 2


def test_explain_discloses_the_throughput_tie_break(capsys) -> None:
    """selection.py orders candidates within 0.01 of the top score by throughput, so the
    `combined` column can be out of order; the output then names the tie-break."""
    main([*_BASE, "--preset", "performance", "--top-k", "5"])
    out = capsys.readouterr().out

    combined = [float(r[-1]) for r in _rows(out)]
    if combined != sorted(combined, reverse=True):
        assert "tie-break" in out
