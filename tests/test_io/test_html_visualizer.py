"""Tests for the HTML report generator (interface.html_visualizer.generate_html_report).

The generator is a pure formatter: it takes Recommendation objects + a
workload_info dict and emits an HTML document. Each test therefore pins a
*transform contract* whose expected output is derived independently from the
inputs (the exact title format string, Python's `.1f`/`.2f` rounding rule,
`str.upper()`, `html.escape()`, the top-1 BEST flag, the `[:5]` cap) rather
than snapshotting whatever markup the template happened to print.

No production code is modified by these tests.
"""

import pytest

from coastline.sdk.io.interface.html_visualizer import generate_html_report
from coastline.sdk.models.recommendation import Recommendation


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def workload_info():
    """A representative workload-info dict matching the keys the report reads."""
    return {
        "llm_model": "mistral-7b-v0.1",
        "method": "lora",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 1024,
        "batch_size": 32,
    }


def _make_rec(
    total_gpus=8,
    gpus_per_node=4,
    number_of_nodes=2,
    throughput=1234.5,
    power=2500.0,
    tokens_per_watt=0.49,
    strategy="multi_objective",
):
    """Build one synthetic Recommendation with power/efficiency in metadata.

    Recommendation validates total_gpus == gpus_per_node * number_of_nodes, so
    callers must keep the layout consistent.
    """
    return Recommendation(
        gpus_per_node=gpus_per_node,
        number_of_nodes=number_of_nodes,
        total_gpus=total_gpus,
        strategy=strategy,
        predicted_throughput=throughput,
        metadata={
            "predicted_power_watts": power,
            "tokens_per_watt": tokens_per_watt,
        },
    )


@pytest.fixture
def recommendations():
    """Two distinct recommendations with hand-checkable, non-colliding values."""
    return [
        # #1 (BEST): 8 GPUs = 4 per node x 2 nodes.
        _make_rec(
            total_gpus=8, gpus_per_node=4, number_of_nodes=2, throughput=1234.56, power=2500.14, tokens_per_watt=0.487
        ),
        # #2: 4 GPUs = 4 per node x 1 node.
        _make_rec(
            total_gpus=4, gpus_per_node=4, number_of_nodes=1, throughput=789.04, power=1300.14, tokens_per_watt=0.616
        ),
    ]


# --------------------------------------------------------------------------- #
# Document contract
# --------------------------------------------------------------------------- #
def test_writes_to_file_and_returns_none(tmp_path, recommendations, workload_info):
    """The function writes the report to disk and returns None (not the string)."""
    out = tmp_path / "report.html"
    ret = generate_html_report(recommendations, workload_info, out)

    # Contract: side-effecting writer, not a string-returning builder.
    assert ret is None
    assert out.exists(), "expected the report file to be created"
    html = out.read_text()
    # Exactly one well-formed document: a single opening doctype and closing tag.
    # (A truncated / doubled write would break this invariant.)
    assert html.count("<!DOCTYPE html>") == 1
    assert html.count("</html>") == 1


# --------------------------------------------------------------------------- #
# Recommendation-card transforms
# --------------------------------------------------------------------------- #
def test_card_title_shows_rank_gpu_count_and_node_layout(tmp_path, recommendations, workload_info):
    """Title is '#<rank>: <total> GPUs (<per_node>x<nodes>)' per the template."""
    out = tmp_path / "report.html"
    generate_html_report(recommendations, workload_info, out)
    html = out.read_text()

    # Oracle = the template literal applied by hand to the fixture inputs:
    #   rec[0] -> rank 1, total 8, layout 4x2 ; rec[1] -> rank 2, total 4, layout 4x1.
    assert "#1: 8 GPUs (4×2)" in html
    assert "#2: 4 GPUs (4×1)" in html


def test_throughput_rendered_to_one_decimal(tmp_path, recommendations, workload_info):
    """Throughput is formatted with `.1f` (round-half-to-even to one decimal)."""
    out = tmp_path / "report.html"
    generate_html_report(recommendations, workload_info, out)
    html = out.read_text()

    # By hand: f"{1234.56:.1f}" -> "1234.6" ; f"{789.04:.1f}" -> "789.0".
    assert "1234.6" in html
    assert "789.0" in html
    # The raw (unrounded) value must not leak -> proves formatting was applied.
    assert "1234.56" not in html


def test_power_uses_one_decimal_efficiency_uses_two(tmp_path, recommendations, workload_info):
    """Power is `.1f`, efficiency is `.2f` — the two use different precisions."""
    out = tmp_path / "report.html"
    generate_html_report(recommendations, workload_info, out)
    html = out.read_text()

    # By hand: power f"{2500.14:.1f}" -> "2500.1" ; f"{1300.14:.1f}" -> "1300.1".
    assert "2500.1" in html
    assert "1300.1" in html
    # By hand: efficiency f"{0.487:.2f}" -> "0.49" ; f"{0.616:.2f}" -> "0.62".
    assert "0.49" in html
    assert "0.62" in html
    # If power had been formatted at .2f it would read "2500.14" — reject that.
    assert "2500.14" not in html


def test_only_top_card_is_flagged_best(tmp_path, recommendations, workload_info):
    """`is_best = (rank == 1)`: exactly one card carries the BEST badge/class."""
    out = tmp_path / "report.html"
    generate_html_report(recommendations, workload_info, out)
    html = out.read_text()

    # Two recs render two cards, but only the first is 'best'.
    assert html.count('class="rec-card') == 2
    assert html.count("rec-card best") == 1
    assert html.count(">BEST</div>") == 1


# --------------------------------------------------------------------------- #
# Workload-info transforms
# --------------------------------------------------------------------------- #
def test_workload_fields_rendered_with_method_uppercased(tmp_path, recommendations, workload_info):
    """Model/GPU rendered verbatim; method is `.upper()`-ed ('lora' -> 'LORA')."""
    out = tmp_path / "report.html"
    generate_html_report(recommendations, workload_info, out)
    html = out.read_text()

    assert "mistral-7b-v0.1" in html  # llm_model, verbatim
    assert "NVIDIA-A100-SXM4-80GB" in html  # gpu_model, verbatim
    # str.upper() contract: "lora" -> "LORA", and the lowercase form is not
    # emitted as the method value.
    assert "LORA" in html


def test_html_metacharacters_in_workload_are_escaped(tmp_path, recommendations):
    """Model/method strings pass through html.escape (no raw markup injection)."""
    out = tmp_path / "escaped.html"
    workload = {"llm_model": "net<a>&b", "method": "q&a"}
    generate_html_report(recommendations, workload, out)
    html = out.read_text()

    # Oracle = html.escape() applied by hand:
    #   "net<a>&b"      -> "net&lt;a&gt;&amp;b"
    #   "q&a".upper()   -> "Q&A" -> "Q&amp;A"
    assert "net&lt;a&gt;&amp;b" in html
    assert "Q&amp;A" in html
    # The un-escaped tag must never appear in the document body.
    assert "net<a>&b" not in html


def test_missing_workload_keys_fall_back_to_na(tmp_path, recommendations):
    """Absent workload_info keys default to 'N/A' (method default 'N/A' upper-cased)."""
    out = tmp_path / "sparse.html"
    generate_html_report(recommendations, {}, out)
    html = out.read_text()

    # .get(..., "N/A"); "N/A".upper() == "N/A".
    assert "N/A" in html
    # Cards do not depend on workload_info, so they still render.
    assert "#1: 8 GPUs (4×2)" in html


# --------------------------------------------------------------------------- #
# Defaults & bounds
# --------------------------------------------------------------------------- #
def test_missing_power_metadata_defaults_to_zero(tmp_path, workload_info):
    """Missing power/efficiency keys default to 0, then format as 0.0 / 0.00."""
    rec = Recommendation(
        gpus_per_node=2,
        number_of_nodes=1,
        total_gpus=2,
        strategy="min-gpu",
        predicted_throughput=100.5,
        metadata={},  # no predicted_power_watts / tokens_per_watt
    )
    out = tmp_path / "no_power.html"
    generate_html_report([rec], workload_info, out)
    html = out.read_text()

    # Throughput unaffected: f"{100.5:.1f}" -> "100.5".
    assert "#1: 2 GPUs (2×1)" in html
    assert "100.5" in html
    # Defaults: f"{0:.1f}" -> "0.0" (power), f"{0:.2f}" -> "0.00" (efficiency).
    assert "0.0" in html
    assert "0.00" in html


def test_empty_recommendation_list_renders_skeleton_without_cards(tmp_path, workload_info):
    """No recommendations => valid document, workload rendered, zero cards/badges."""
    out = tmp_path / "empty.html"
    generate_html_report([], workload_info, out)
    html = out.read_text()

    assert html.count("<!DOCTYPE html>") == 1
    assert html.count("</html>") == 1
    assert "mistral-7b-v0.1" in html  # workload still rendered
    # `class="rec-card` is only emitted per rendered card (the CSS rule is
    # `.rec-card`, without the `class="` prefix), so zero cards => zero matches.
    assert html.count('class="rec-card') == 0
    assert "BEST" not in html


def test_caps_rendered_cards_at_five(tmp_path, workload_info):
    """The report slices recommendations[:5]; extras (#6, #7) are dropped."""
    recs = [
        _make_rec(total_gpus=i, gpus_per_node=i, number_of_nodes=1, throughput=float(i))
        for i in range(1, 8)  # 7 recommendations
    ]
    out = tmp_path / "capped.html"
    generate_html_report(recs, workload_info, out)
    html = out.read_text()

    # Exactly five cards; #5 present, #6/#7 absent (slice upper bound).
    assert html.count('class="rec-card') == 5
    assert "#5: 5 GPUs (5×1)" in html
    assert "#6: 6 GPUs (6×1)" not in html
    assert "#7: 7 GPUs (7×1)" not in html
    # The single BEST flag survives regardless of list length.
    assert html.count("rec-card best") == 1
