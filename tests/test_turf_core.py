"""Unit tests for the pure TURF maths and data-loading helpers."""

from __future__ import annotations

import io
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import turf_core as tc


# ── helpers ──────────────────────────────────────────────────────────────────
def random_frame(n=60, p=8, seed=0, density=0.25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = (rng.random((n, p)) < density).astype(np.int8)
    return pd.DataFrame(data, columns=[f"I{i}" for i in range(p)])


def brute_force_best(df: pd.DataFrame, k: int, include=(), exclude=()):
    cands = [c for c in df.columns if c not in include and c not in exclude]
    best = (-1, -1)
    for extra in combinations(cands, k - len(include)):
        cols = [*include, *extra]
        reach = int((df[cols].sum(axis=1) > 0).sum())
        freq = int(df[cols].to_numpy().sum())
        best = max(best, (reach, freq))
    return best


def combo_set(label: str) -> frozenset[str]:
    return frozenset(label.split(" + "))


# ── column heuristics ────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["id", "ID", "respondent_id", "RespondentID", "userId",
                                  "customer_id", "Timestamp", "date", "index", "resp id"])
def test_is_id_column_true(name):
    assert tc.is_id_column(name)


@pytest.mark.parametrize("name", ["humidity", "lipid", "Item_01", "Q1_TV", "Radio", "idea", 5])
def test_is_id_column_false(name):
    assert not tc.is_id_column(name)


# ── cleaning ─────────────────────────────────────────────────────────────────
def test_clean_accepts_bool_columns():
    df = pd.DataFrame({"a": [True, False, True], "b": [False, False, True]})
    result = tc.clean_binary_data(df, ["a", "b"])
    assert result.error is None
    assert result.data.to_numpy().tolist() == [[1, 0], [0, 0], [1, 1]]
    assert result.data.dtypes.eq(np.int8).all()


def test_clean_accepts_yes_no_text():
    df = pd.DataFrame({"a": ["Yes", "no", " Y "], "b": ["1", "0", "1"]})
    result = tc.clean_binary_data(df, ["a", "b"])
    assert result.error is None
    assert result.data.to_numpy().tolist() == [[1, 1], [0, 0], [1, 1]]


def test_clean_missing_values_are_zero_and_reported():
    df = pd.DataFrame({"a": [1, np.nan, 0], "b": [1, 1, None]})
    result = tc.clean_binary_data(df, ["a", "b"])
    assert result.data.to_numpy().tolist() == [[1, 1], [0, 1], [0, 0]]
    assert any("missing" in w for w in result.warnings)


def test_clean_non_binary_values_are_converted_and_reported():
    df = pd.DataFrame({"a": [0, 2, 5], "b": [-1, 1, 0]})
    result = tc.clean_binary_data(df, ["a", "b"])
    assert result.data.to_numpy().tolist() == [[0, 0], [1, 1], [1, 0]]
    assert any("Non-binary" in w for w in result.warnings)


def test_clean_drops_non_numeric_and_requires_two_columns():
    df = pd.DataFrame({"a": [1, 0], "name": ["x", "y"], "b": [0, 1]})
    ok = tc.clean_binary_data(df, ["a", "name", "b"])
    assert list(ok.data.columns) == ["a", "b"]
    assert any("Non-numeric" in w for w in ok.warnings)

    bad = tc.clean_binary_data(df, ["a", "name"])
    assert bad.data is None and bad.error


def test_clean_rejects_empty_data():
    df = pd.DataFrame({"a": [], "b": []})
    assert tc.clean_binary_data(df, ["a", "b"]).error


def test_is_binary_like():
    assert tc.is_binary_like(pd.Series([0, 1, 1, None]))
    assert tc.is_binary_like(pd.Series([True, False]))
    assert tc.is_binary_like(pd.Series(["yes", "no"]))
    assert not tc.is_binary_like(pd.Series([1, 2, 3]))
    assert not tc.is_binary_like(pd.Series(["a", "b"]))
    assert not tc.is_binary_like(pd.Series([None, None]))


def test_normalize_columns_makes_unique_strings():
    df = pd.DataFrame([[1, 2, 3, 4]], columns=[1, " a ", "a", ""])
    assert list(tc.normalize_columns(df).columns) == ["1", "a", "a_2", "Column_4"]


# ── file reading ─────────────────────────────────────────────────────────────
def test_read_csv_variants():
    expected = [[1, 0], [0, 1]]
    plain = b"a,b\n1,0\n0,1\n"
    bom = b"\xef\xbb\xbfa,b\n1,0\n0,1\n"
    semicolon = b"a;b\n1;0\n0;1\n"
    latin1 = "caf\xe9,b\n1,0\n0,1\n".encode("latin-1")
    for raw in (plain, bom, semicolon, latin1):
        df = tc.read_table(raw, "data.csv")
        assert df.to_numpy().tolist() == expected
    assert list(tc.read_table(bom, "data.csv").columns) == ["a", "b"]
    assert tc.read_table(latin1, "data.csv").columns[0] == "caf\xe9"


def test_read_excel_roundtrip_and_sheets():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"a": [1, 0], "b": [0, 1]}).to_excel(writer, sheet_name="First", index=False)
        pd.DataFrame({"c": [1, 1], "d": [0, 0]}).to_excel(writer, sheet_name="Second", index=False)
    data = buffer.getvalue()
    assert tc.list_excel_sheets(data) == ["First", "Second"]
    assert list(tc.read_table(data, "x.xlsx").columns) == ["a", "b"]
    assert list(tc.read_table(data, "x.xlsx", "Second").columns) == ["c", "d"]


def test_read_table_drops_blank_rows_and_columns():
    buffer = io.BytesIO()
    frame = pd.DataFrame({"a": [1, None, 0], "b": [0, None, 1], "empty": [None, None, None]})
    frame.to_excel(buffer, index=False)
    df = tc.read_table(buffer.getvalue(), "x.xlsx")
    assert df.shape == (2, 2)


def test_read_table_errors_are_data_errors():
    with pytest.raises(tc.DataError):
        tc.read_table(b"", "x.csv")
    with pytest.raises(tc.DataError):
        tc.read_table(b"hello", "x.txt")
    with pytest.raises(tc.DataError):
        tc.read_table(b"not an excel file", "x.xlsx")
    with pytest.raises(tc.DataError):
        tc.read_table(b"a,b\n", "x.csv")


# ── individual reach ─────────────────────────────────────────────────────────
def test_individual_reach_matches_column_sums_and_respects_exclude():
    df = random_frame()
    out = tc.individual_reach(df, must_exclude=("I0",))
    assert "I0" not in set(out["Item"])
    for _, row in out.iterrows():
        assert row["Reach (Count)"] == int(df[row["Item"]].sum())
        assert row["Reach (%)"] == round(df[row["Item"]].mean() * 100, 2)
    assert out["Reach (Count)"].is_monotonic_decreasing


# ── exhaustive search ────────────────────────────────────────────────────────
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_exhaustive_matches_brute_force(k):
    df = random_frame(seed=k)
    result = tc.calculate_turf(df, k)
    assert not result.is_sampled and result.coverage_pct == 100.0
    assert result.theoretical_combos == len(result.results)
    top = result.results.iloc[0]
    assert (top["Reach (Count)"], top["Frequency"]) == brute_force_best(df, k)


def test_exhaustive_rows_are_correct_and_sorted():
    df = random_frame(n=40, p=6, seed=3)
    result = tc.calculate_turf(df, 3).results
    for _, row in result.iterrows():
        cols = row["Combination"].split(" + ")
        assert len(cols) == 3
        assert row["Reach (Count)"] == int((df[cols].sum(axis=1) > 0).sum())
        assert row["Frequency"] == int(df[cols].to_numpy().sum())
        assert row["Reach (%)"] == round(row["Reach (Count)"] / 40 * 100, 2)
    keys = list(zip(result["Reach (Count)"], result["Frequency"], strict=True))
    assert keys == sorted(keys, reverse=True)


def test_constraints_are_respected():
    df = random_frame()
    result = tc.calculate_turf(df, 4, must_include=("I1",), must_exclude=("I2", "I3"))
    assert not result.is_sampled
    for label in result.results["Combination"]:
        names = label.split(" + ")
        assert names[0] == "I1"
        assert not {"I2", "I3"} & set(names)
        assert len(names) == 4
    top = result.results.iloc[0]
    assert (top["Reach (Count)"], top["Frequency"]) == brute_force_best(
        df, 4, include=["I1"], exclude=["I2", "I3"]
    )


def test_infeasible_k_returns_empty_results():
    df = random_frame(p=5)
    assert tc.calculate_turf(df, 1, must_include=("I0", "I1")).results.empty  # k < locked items
    assert tc.calculate_turf(df, 6).results.empty                              # k > available
    assert tc.calculate_turf(df, 5, must_exclude=("I0",)).results.empty


def test_k_equal_to_locked_items_evaluates_one_portfolio():
    df = random_frame(p=5)
    result = tc.calculate_turf(df, 2, must_include=("I0", "I1"))
    assert len(result.results) == 1
    assert result.results.iloc[0]["Combination"] == "I0 + I1"


def test_invalid_constraints_raise():
    df = random_frame()
    with pytest.raises(ValueError, match="Unknown"):
        tc.calculate_turf(df, 2, must_include=("nope",))
    with pytest.raises(ValueError, match="both"):
        tc.calculate_turf(df, 2, must_include=("I0",), must_exclude=("I0",))
    with pytest.raises(ValueError, match="no respondents"):
        tc.calculate_turf(df.iloc[0:0], 2)


# ── sampling ─────────────────────────────────────────────────────────────────
def test_sampling_is_flagged_unique_and_includes_greedy():
    df = random_frame(n=80, p=10, seed=7)
    result = tc.calculate_turf(df, 4, max_combinations=50)   # C(10,4)=210 > 50
    assert result.is_sampled
    assert result.theoretical_combos == 210
    assert 0 < result.coverage_pct < 100
    sets = [combo_set(c) for c in result.results["Combination"]]
    assert len(sets) == len(set(sets)), "duplicate portfolios in sampled results"

    greedy = tc.greedy_turf(df, 4)
    assert frozenset(greedy.portfolio) in set(sets)
    greedy_reach = int((df[greedy.portfolio].sum(axis=1) > 0).sum())
    assert result.results.iloc[0]["Reach (Count)"] >= greedy_reach


def test_sampling_is_deterministic():
    df = random_frame(n=80, p=10, seed=7)
    a = tc.calculate_turf(df, 4, max_combinations=50).results
    b = tc.calculate_turf(df, 4, max_combinations=50).results
    pd.testing.assert_frame_equal(a, b)


def test_sampling_just_above_threshold_terminates():
    df = random_frame(n=50, p=10, seed=2)
    result = tc.calculate_turf(df, 5, max_combinations=251)  # C(10,5)=252
    assert result.is_sampled


# ── greedy ───────────────────────────────────────────────────────────────────
def test_greedy_curve_is_consistent():
    df = random_frame(seed=11)
    g = tc.greedy_turf(df, 5)
    assert len(g.portfolio) == 5
    assert len(g.reach_curve) == 6 and len(g.marginal_gains) == 5
    assert g.reach_curve[0] == 0.0
    assert g.reach_curve == sorted(g.reach_curve)
    final = int((df[g.portfolio].sum(axis=1) > 0).sum())
    assert g.reach_curve[-1] == round(final / len(df) * 100, 2)
    assert sum(g.marginal_counts) == final
    assert all(c >= 0 for c in g.marginal_counts)


def test_greedy_picks_largest_gain_with_frequency_tiebreak():
    df = pd.DataFrame({
        "A": [1, 1, 1, 0, 0, 0],
        "B": [1, 1, 0, 0, 0, 0],   # subset of A
        "C": [0, 0, 0, 1, 1, 0],
        "D": [0, 0, 0, 1, 0, 1],
    })
    g = tc.greedy_turf(df, 3)
    assert g.portfolio[0] == "A"
    # C and D both add 2 new respondents and have equal frequency: column order decides.
    assert g.portfolio[1] == "C"
    assert g.marginal_counts[:2] == [3, 2]


def test_greedy_with_must_include():
    df = random_frame(seed=5)
    g = tc.greedy_turf(df, 4, must_include=("I3",), must_exclude=("I0",))
    assert g.portfolio[0] == "I3"
    assert "I0" not in g.portfolio
    assert len(g.portfolio) == 4
    assert len(g.reach_curve) == 1 + 3 and len(g.marginal_gains) == 3
    baseline = round(df["I3"].sum() / len(df) * 100, 2)
    assert g.reach_curve[0] == baseline


def test_greedy_never_exceeds_available_items():
    df = random_frame(p=4)
    g = tc.greedy_turf(df, 10, must_exclude=("I0",))
    assert len(g.portfolio) == 3


# ── optimal by size ──────────────────────────────────────────────────────────
def test_optimal_by_size_exact_matches_brute_force():
    df = random_frame(seed=4, p=7)
    out = tc.optimal_by_size(df, 4)
    assert list(out["Size"]) == [1, 2, 3, 4]
    assert set(out["Method"]) == {"Exact"}
    for _, row in out.iterrows():
        assert row["Reach (Count)"] == brute_force_best(df, row["Size"])[0]
    assert out["Reach (%)"].is_monotonic_increasing
    assert out["Incremental Reach (%)"].iloc[0] == out["Reach (%)"].iloc[0]


def test_optimal_by_size_never_decreases_when_approximating():
    for seed in range(5):
        df = random_frame(n=100, p=10, seed=seed, density=0.2)
        out = tc.optimal_by_size(df, 8, max_combinations=20)
        assert (out["Method"] == "Greedy approx.").any()
        assert (out["Incremental Reach (%)"] >= 0).all(), out
        assert out["Reach (%)"].is_monotonic_increasing
        for _, row in out.iterrows():
            cols = row["Combination"].split(" + ")
            assert len(cols) == row["Size"]
            assert row["Reach (Count)"] == int((df[cols].sum(axis=1) > 0).sum())


def test_optimal_by_size_with_constraints():
    df = random_frame(seed=8, p=7)
    out = tc.optimal_by_size(df, 5, must_include=("I1", "I2"), must_exclude=("I0",))
    assert list(out["Size"]) == [2, 3, 4, 5]
    for label in out["Combination"]:
        names = label.split(" + ")
        assert names[:2] == ["I1", "I2"] and "I0" not in names


def test_optimal_by_size_stops_when_items_run_out():
    df = random_frame(p=4)
    assert list(tc.optimal_by_size(df, 12)["Size"]) == [1, 2, 3, 4]


# ── helpers ──────────────────────────────────────────────────────────────────
def test_diminishing_returns_size():
    assert tc.diminishing_returns_size([1, 2, 3, 4], [40.0, 20.0, 1.5, 5.0]) == 2
    assert tc.diminishing_returns_size([1, 2, 3], [40.0, 20.0, 10.0]) is None
    # a tiny first gain is never the cut-off, so the answer is never "0 items"
    assert tc.diminishing_returns_size([1, 2], [0.5, 0.1]) == 1


def test_simulate_data_is_deterministic_and_binary():
    a = tc.simulate_data(100, 6)
    b = tc.simulate_data(100, 6)
    pd.testing.assert_frame_equal(a, b)
    assert a.shape == (100, 6)
    assert set(np.unique(a.to_numpy())) <= {0, 1}


def test_fingerprint_changes_with_data_and_settings():
    df = random_frame()
    base = tc.fingerprint(df, 3)
    assert base == tc.fingerprint(df.copy(), 3)
    assert base != tc.fingerprint(df, 4)
    changed = df.copy()
    changed.iloc[0, 0] = 1 - changed.iloc[0, 0]
    assert base != tc.fingerprint(changed, 3)
