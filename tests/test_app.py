"""End-to-end tests that drive the real Streamlit script headlessly."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


needs_uploader = pytest.mark.skipif(
    not hasattr(AppTest, "file_uploader"),
    reason="this Streamlit version's AppTest cannot simulate file uploads",
)


def new_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120).run()


def widget(collection, label_part: str):
    matches = [w for w in collection if label_part in w.label]
    assert matches, f"no widget with label containing {label_part!r}"
    return matches[0]


def to_upload_mode(at: AppTest) -> AppTest:
    at.radio[0].set_value("Upload Own Data").run()
    return at


def csv_upload(at: AppTest, text: str, name: str = "data.csv") -> AppTest:
    at.file_uploader[0].set_value((name, text.encode(), "text/csv")).run()
    return at


def plotly_specs(at: AppTest) -> list[dict]:
    return [json.loads(el.proto.spec) for el in at.get("plotly_chart")]


def reach_curve_spec(at: AppTest) -> dict:
    for spec in plotly_specs(at):
        if spec["layout"].get("title", {}).get("text") == "Reach Curve (Greedy Build-Up)":
            return spec
    raise AssertionError("reach curve chart not found")


def survey_csv(n=40, extra_header="respondent_id", seed=0, cols=("TV", "Radio", "Web", "Print")) -> str:
    rng = np.random.default_rng(seed)
    lines = [",".join([extra_header, *cols])]
    for i in range(n):
        row = (rng.random(len(cols)) < 0.3).astype(int)
        lines.append(",".join([str(i + 1), *map(str, row)]))
    return "\n".join(lines) + "\n"


# ── simulated data ───────────────────────────────────────────────────────────
def test_loads_without_errors():
    at = new_app()
    assert not at.exception
    assert at.button[0].label.endswith("Run TURF Analysis")


def test_default_run_produces_results_without_warnings_about_deprecations(caplog):
    caplog.set_level(logging.WARNING)
    at = new_app()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.tabs) == 5
    assert at.session_state["turf_results"].shape[0] > 0
    assert "use_container_width" not in caplog.text


def test_slider_with_a_single_possible_value_does_not_crash():
    """5 items with 4 locked leaves exactly one valid k; st.slider raises if min == max."""
    at = new_app()
    widget(at.slider, "Number of Items").set_value(5).run()
    widget(at.multiselect, "Must-Include").set_value(["Item_01", "Item_02", "Item_03", "Item_04"]).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["turf_k"] == 5


def test_constraints_that_leave_too_few_items_show_an_error():
    at = new_app()
    widget(at.slider, "Number of Items").set_value(5).run()
    all_five = ["Item_01", "Item_02", "Item_03", "Item_04", "Item_05"]
    widget(at.multiselect, "Must-Include").set_value(all_five).run()
    assert not at.exception
    assert any("Too many constraints" in e.value for e in at.error)
    assert not at.button


def test_slider_range_never_exceeds_available_items():
    at = new_app()
    widget(at.slider, "Number of Items").set_value(6).run()
    widget(at.multiselect, "Must-Exclude").set_value(["Item_01", "Item_02", "Item_03"]).run()
    k = widget(at.slider, "Portfolio Size")
    assert k.max == 3


def test_must_include_greedy_table_and_curve_are_correct(caplog):
    caplog.set_level(logging.WARNING)
    at = new_app()
    widget(at.multiselect, "Must-Include").set_value(["Item_01"]).run()
    at.button[0].click().run()
    assert not at.exception
    assert "Arrow" not in caplog.text  # the old table mixed "—" strings with numbers

    greedy = at.session_state["turf_greedy"]
    spec = reach_curve_spec(at)
    line, bars = spec["data"]
    # x is the portfolio size; the first point is the must-include set (size 1)
    assert line["x"][0] == 1
    assert len(line["x"]) == len(greedy.reach_curve)
    assert bars["x"] == line["x"][1:]
    assert len(bars["y"]) == len(greedy.marginal_gains)
    # tick labels: the starting set, then the greedy items — not the must-include item again
    assert spec["layout"]["xaxis"]["ticktext"] == ["Must-include", *greedy.portfolio[1:]]
    # the marginal-gain axis must be tall enough for the biggest bar (it used to be capped at 30)
    assert spec["layout"]["yaxis2"]["range"][1] >= max(greedy.marginal_gains)


def test_marginal_axis_is_not_clipped_when_first_gain_is_large():
    at = new_app()
    at.button[0].click().run()
    greedy = at.session_state["turf_greedy"]
    assert max(greedy.marginal_gains) > 0
    assert reach_curve_spec(at)["layout"]["yaxis2"]["range"][1] >= max(greedy.marginal_gains)


def test_stale_banner_after_changing_configuration():
    at = new_app()
    at.button[0].click().run()
    assert not any("has changed" in w.value for w in at.warning)
    widget(at.slider, "Portfolio Size").set_value(4).run()
    assert any("has changed" in w.value for w in at.warning)


# ── uploads ──────────────────────────────────────────────────────────────────
@needs_uploader
def test_upload_with_id_column_and_underscore_name():
    at = to_upload_mode(new_app())
    csv_upload(at, survey_csv())
    assert not at.exception
    selected = widget(at.multiselect, "Select columns").value
    assert "respondent_id" not in selected
    assert selected == ["TV", "Radio", "Web", "Print"]
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["turf_results"].shape[0] > 0


@needs_uploader
def test_upload_with_exactly_two_items_does_not_crash():
    at = to_upload_mode(new_app())
    csv_upload(at, survey_csv(cols=("TV", "Radio")))
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["turf_k"] == 2


@needs_uploader
def test_upload_with_boolean_and_yes_no_columns():
    text = "id,A,B,C\n1,True,yes,1\n2,False,no,0\n3,True,no,1\n4,False,yes,0\n"
    at = to_upload_mode(new_app())
    csv_upload(at, text)
    assert not at.exception
    assert widget(at.multiselect, "Select columns").value == ["A", "B", "C"]
    at.button[0].click().run()
    assert not at.exception
    # respondent 2 selected nothing, so 3 of 4 respondents is the best possible reach
    assert at.session_state["turf_results"].iloc[0]["Reach (%)"] == 75.0


@needs_uploader
def test_numeric_headers_are_handled():
    text = "1,2,3\n1,0,1\n0,1,0\n1,1,0\n"
    at = to_upload_mode(new_app())
    csv_upload(at, text)
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception


@needs_uploader
def test_bad_uploads_show_errors_instead_of_crashing():
    at = to_upload_mode(new_app())
    csv_upload(at, "a,b\n")
    assert not at.exception and at.error
    csv_upload(at, "name,city\nx,y\nz,w\n")
    assert not at.exception and at.error


@needs_uploader
def test_results_are_flagged_stale_when_data_changes_but_shape_does_not():
    at = to_upload_mode(new_app())
    csv_upload(at, survey_csv(seed=1))
    at.button[0].click().run()
    assert not any("has changed" in w.value for w in at.warning)
    csv_upload(at, survey_csv(seed=2))   # same columns, same row count, different answers
    assert any("has changed" in w.value for w in at.warning)


@needs_uploader
def test_results_are_cleared_when_the_file_is_removed():
    at = to_upload_mode(new_app())
    csv_upload(at, survey_csv())
    at.button[0].click().run()
    assert "turf_results" in at.session_state
    at.file_uploader[0].set_value(None).run()
    assert "turf_results" not in at.session_state
    assert not at.exception


@pytest.mark.parametrize("k_total", [4, 12])
def test_runs_with_sampled_search_space(k_total):
    """20 items at k=12 is far beyond the exhaustive limit and exercises the sampling path."""
    at = new_app()
    widget(at.slider, "Number of Items").set_value(20).run()
    widget(at.slider, "Portfolio Size").set_value(k_total).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["turf_results"].shape[0] > 0
