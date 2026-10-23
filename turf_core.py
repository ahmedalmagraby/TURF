"""Core TURF (Total Unduplicated Reach and Frequency) computations.

This module has no Streamlit dependency, so the maths can be unit-tested and reused
outside the web app.

Terminology
-----------
item          one column of the binary respondent x item matrix
portfolio     a set of items
reach         share of respondents who selected at least one item of a portfolio
frequency     total number of selections across the items of a portfolio
must-include  items locked into every portfolio
must-exclude  items that are never considered
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections.abc import Hashable, Iterable, Sequence
from itertools import chain, combinations
from typing import NamedTuple

import numpy as np
import pandas as pd

# Largest search space that is enumerated exhaustively. Beyond this a random sample of
# combinations (plus the greedy solution) is evaluated instead.
MAX_COMBINATIONS = 15_000
# When sampling, never try to draw more than this share of the whole space, otherwise
# drawing *unique* combinations becomes very slow (coupon-collector problem).
SAMPLE_SPACE_CAP = 0.75
RNG_SEED = 42
# Upper bound on respondents x combinations boolean cells held in memory at once.
_CHUNK_CELLS = 20_000_000

RESULT_COLUMNS = ["Combination", "Reach (%)", "Reach (Count)", "Frequency"]

EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")
SUPPORTED_SUFFIXES = (".csv", *EXCEL_SUFFIXES)

_ID_TOKENS = frozenset({
    "id", "uid", "uuid", "userid", "respid", "respondentid", "customerid",
    "index", "respondent", "user", "customer", "demographic", "timestamp", "date",
})
_TRUE_STRINGS = frozenset({"1", "y", "yes", "true", "t"})
_FALSE_STRINGS = frozenset({"0", "n", "no", "false", "f"})
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")


class DataError(ValueError):
    """A problem with user-supplied data whose message is safe to show to the user."""


class TurfSearch(NamedTuple):
    results: pd.DataFrame
    is_sampled: bool
    theoretical_combos: int
    coverage_pct: float


class GreedyResult(NamedTuple):
    portfolio: list[str]           # must-include items first, then greedy picks in order
    reach_curve: list[float]       # reach (%) of the baseline, then after each greedy pick
    marginal_gains: list[float]    # extra reach (%) contributed by each greedy pick
    marginal_counts: list[int]     # extra respondents contributed by each greedy pick


class CleanResult(NamedTuple):
    data: pd.DataFrame | None
    warnings: list[str]
    error: str | None


# ══════════════════════════════════════════════════════════════════════════════
#  Loading and cleaning
# ══════════════════════════════════════════════════════════════════════════════
def is_id_column(name: Hashable) -> bool:
    """Heuristic: does this column name look like an ID / metadata column?

    The name is split into tokens (on punctuation and camelCase), so ``respondent_id``,
    ``RespondentID`` and ``userId`` are caught while ``humidity`` or ``lipid`` are not.
    """
    text = _CAMEL_BOUNDARY.sub(" ", str(name))
    tokens = {t.lower() for t in _NON_ALNUM.split(text) if t}
    return bool(tokens & _ID_TOKENS)


def _parse_scalar(value: object) -> float | None:
    """Parse one cell to a float; ``None`` if it cannot be interpreted."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return 1.0
        if text in _FALSE_STRINGS:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return None
    if isinstance(value, (bool, np.bool_, int, float, np.number)):
        return float(value)
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series | None:
    """Return ``series`` as float64 (NaN = missing), or ``None`` if it is not interpretable.

    Accepts numeric and boolean columns as well as text columns holding numbers or
    yes/no/true/false style answers.
    """
    try:
        if pd.api.types.is_bool_dtype(series):
            return series.astype("float64")
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
            return None
        if pd.api.types.is_numeric_dtype(series):
            return series.astype("float64")
    except (TypeError, ValueError):
        return None

    values: list[float] = []
    for cell in series.astype("object"):
        if cell is None or cell is pd.NA or (isinstance(cell, float) and math.isnan(cell)):
            values.append(math.nan)
            continue
        if isinstance(cell, str) and not cell.strip():
            values.append(math.nan)
            continue
        parsed = _parse_scalar(cell)
        if parsed is None:
            return None
        values.append(parsed)
    if all(math.isnan(v) for v in values):
        return None
    return pd.Series(values, index=series.index, dtype="float64")


def is_binary_like(series: pd.Series) -> bool:
    """True if every non-missing value is 0/1 (or a yes/no-style equivalent)."""
    numeric = _coerce_numeric(series)
    if numeric is None:
        return False
    values = numeric.dropna()
    return (not values.empty) and bool(values.isin([0.0, 1.0]).all())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names unique, non-empty strings (Excel files often have numeric headers)."""
    used: set[str] = set()
    names: list[str] = []
    for position, column in enumerate(df.columns, start=1):
        base = str(column).strip() or f"Column_{position}"
        name, suffix = base, 1
        while name in used:
            suffix += 1
            name = f"{base}_{suffix}"
        used.add(name)
        names.append(name)
    return df.set_axis(names, axis=1)


def _read_csv(data: bytes) -> pd.DataFrame:
    # utf-8-sig strips the BOM that Excel adds; latin-1 can decode any byte sequence.
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError:
            continue
        if df.shape[1] == 1:
            # Possibly a semicolon- or tab-separated file: let pandas sniff the delimiter.
            try:
                sniffed = pd.read_csv(io.BytesIO(data), encoding=encoding, sep=None, engine="python")
                if sniffed.shape[1] > 1:
                    df = sniffed
            except Exception:  # noqa: BLE001 - sniffing is best effort
                pass
        return df
    raise DataError("The file's text encoding could not be determined. Save it as UTF-8 and retry.")


def list_excel_sheets(data: bytes) -> list[str]:
    """Worksheet names of an Excel workbook."""
    try:
        with pd.ExcelFile(io.BytesIO(data)) as workbook:
            return [str(name) for name in workbook.sheet_names]
    except Exception as exc:  # noqa: BLE001
        raise DataError(f"Could not open the Excel file: {exc}") from exc


def read_table(data: bytes, filename: str, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Parse an uploaded CSV / Excel file into a DataFrame with clean, unique column names.

    Entirely empty rows and columns (common at the end of Excel sheets) are dropped so
    they cannot be mistaken for respondents who selected nothing.
    """
    if not data:
        raise DataError("The uploaded file is empty.")
    lower = filename.lower()
    try:
        if lower.endswith(".csv"):
            df = _read_csv(data)
        elif lower.endswith(EXCEL_SUFFIXES):
            df = pd.read_excel(io.BytesIO(data), sheet_name=0 if sheet_name is None else sheet_name)
        else:
            raise DataError(
                f"Unsupported file type. Please upload one of: {', '.join(SUPPORTED_SUFFIXES)}"
            )
    except DataError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataError(f"Could not read '{filename}': {exc}") from exc

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        raise DataError("The file contains no data.")
    return normalize_columns(df).reset_index(drop=True)


def clean_binary_data(df: pd.DataFrame, columns: Sequence[str]) -> CleanResult:
    """Turn the chosen columns into a clean 0/1 int8 respondent x item matrix.

    Handles bool columns, yes/no style text and missing values (treated as 0), and converts
    other numeric values with ``> 0 -> 1``. Anything that was changed is reported in
    ``warnings`` so the caller can tell the user.
    """
    if len(df) == 0:
        return CleanResult(None, [], "The file contains no respondent rows.")
    columns = list(columns)
    absent = [c for c in columns if c not in df.columns]
    if absent:
        return CleanResult(None, [], f"Unknown column(s): {', '.join(map(str, absent))}")

    warnings: list[str] = []
    converted: dict[str, pd.Series] = {}
    dropped: list[str] = []
    for column in columns:
        numeric = _coerce_numeric(df[column])
        if numeric is None:
            dropped.append(str(column))
        else:
            converted[column] = numeric
    if dropped:
        warnings.append(f"Non-numeric columns excluded: {', '.join(dropped)}")
    if len(converted) < 2:
        return CleanResult(
            None, warnings,
            "At least two usable item columns (0/1, yes/no or other numeric values) are needed.",
        )

    frame = pd.DataFrame(converted, index=df.index)
    missing_cells = int(frame.isna().sum().sum())
    if missing_cells:
        warnings.append(f"{missing_cells:,} missing value(s) found and treated as 0 (not selected).")

    distinct = np.unique(frame.to_numpy().ravel())
    non_binary = [v for v in distinct if not np.isnan(v) and v not in (0.0, 1.0)]
    if non_binary:
        sample = ", ".join(f"{v:g}" for v in non_binary[:10]) + (", …" if len(non_binary) > 10 else "")
        warnings.append(
            f"Non-binary values detected ({sample}). Converting values > 0 to 1 and all others to 0."
        )

    clean = (frame.fillna(0.0) > 0).astype(np.int8).reset_index(drop=True)
    unselected = [str(c) for c in clean.columns if not clean[c].any()]
    if unselected:
        warnings.append(f"Items that no respondent selected: {', '.join(unselected)}")
    return CleanResult(clean, warnings, None)


def simulate_data(n_respondents: int = 200, n_items: int = 12, seed: int = RNG_SEED) -> pd.DataFrame:
    """Synthetic binary data with a different selection probability per item."""
    rng = np.random.default_rng(seed)
    probabilities = rng.uniform(0.15, 0.45, n_items)
    data = rng.binomial(1, probabilities, (n_respondents, n_items)).astype(np.int8)
    return pd.DataFrame(data, columns=[f"Item_{i + 1:02d}" for i in range(n_items)])


def fingerprint(df: pd.DataFrame, *parts: object) -> str:
    """Stable hash of the data and any extra settings, used to detect stale results."""
    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, df.columns)).encode())
    digest.update(repr(df.shape).encode())
    digest.update(np.ascontiguousarray(df.to_numpy()).tobytes())
    for part in parts:
        digest.update(b"\x00" + repr(part).encode())
    return digest.hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
#  TURF maths
# ══════════════════════════════════════════════════════════════════════════════
class _Problem(NamedTuple):
    items: list[str]
    must_include: list[str]
    candidates: list[str]
    matrix: np.ndarray             # bool, respondents x items
    index: dict[str, int]          # item -> column in ``matrix``


def _prepare(df: pd.DataFrame, must_include: Iterable[str], must_exclude: Iterable[str]) -> _Problem:
    if len(df) == 0:
        raise ValueError("The data contains no respondents.")
    if df.shape[1] == 0:
        raise ValueError("The data contains no items.")
    if df.columns.has_duplicates:
        raise ValueError("Item names must be unique.")

    items = list(df.columns)
    known = set(items)
    include = list(dict.fromkeys(must_include))
    exclude = set(must_exclude)
    unknown = [x for x in (*include, *exclude) if x not in known]
    if unknown:
        raise ValueError(f"Unknown item(s): {', '.join(map(str, unknown))}")
    clash = [x for x in include if x in exclude]
    if clash:
        raise ValueError(
            f"Items cannot be both must-include and must-exclude: {', '.join(map(str, clash))}"
        )
    locked = set(include)
    candidates = [it for it in items if it not in exclude and it not in locked]
    return _Problem(
        items=items,
        must_include=include,
        candidates=candidates,
        matrix=df.to_numpy() > 0,
        index={it: i for i, it in enumerate(items)},
    )


def _base(problem: _Problem) -> tuple[np.ndarray, int]:
    """Respondents reached (and selections made) by the must-include items alone."""
    n = problem.matrix.shape[0]
    if not problem.must_include:
        return np.zeros(n, dtype=bool), 0
    columns = problem.matrix[:, [problem.index[it] for it in problem.must_include]]
    return columns.any(axis=1), int(columns.sum())


def _all_combinations(n_candidates: int, extra_k: int) -> np.ndarray:
    total = math.comb(n_candidates, extra_k)
    flat = np.fromiter(
        chain.from_iterable(combinations(range(n_candidates), extra_k)),
        dtype=np.intp,
        count=total * extra_k,
    )
    return flat.reshape(total, extra_k)


def _sample_combinations(n_candidates: int, extra_k: int, target: int, seed: int) -> set[tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()
    while len(seen) < target:
        seen.add(tuple(sorted(rng.choice(n_candidates, extra_k, replace=False).tolist())))
    return seen


def _evaluate(
    cand_matrix: np.ndarray,
    base_reached: np.ndarray,
    base_freq: int,
    combos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reach count and frequency for every combination (rows of ``combos``)."""
    n = cand_matrix.shape[0]
    m, extra_k = combos.shape
    column_freq = cand_matrix.sum(axis=0, dtype=np.int64)
    freqs = base_freq + column_freq[combos].sum(axis=1)
    counts = np.empty(m, dtype=np.int64)
    chunk = max(1, min(4096, _CHUNK_CELLS // max(n, 1)))
    for start in range(0, m, chunk):
        block = combos[start:start + chunk]
        reached = np.repeat(base_reached[:, None], block.shape[0], axis=1)
        for j in range(extra_k):
            reached |= cand_matrix[:, block[:, j]]
        counts[start:start + block.shape[0]] = reached.sum(axis=0)
    return counts, freqs


def _best_index(counts: np.ndarray, freqs: np.ndarray) -> int:
    """Index of the highest reach; ties go to higher frequency, then to enumeration order."""
    return int(np.lexsort((-np.arange(len(counts)), freqs, counts))[-1])


def _pct(count: int | float, n: int) -> float:
    return round(float(count) / n * 100, 2)


def calculate_turf(
    df: pd.DataFrame,
    k: int,
    must_include: Sequence[str] = (),
    must_exclude: Sequence[str] = (),
    max_combinations: int = MAX_COMBINATIONS,
    seed: int = RNG_SEED,
) -> TurfSearch:
    """Evaluate portfolios of exactly ``k`` items (including must-include items).

    Exhaustive when the search space has at most ``max_combinations`` portfolios. Otherwise
    a random sample is evaluated and the greedy portfolio is always added to it, so the
    reported best is never worse than the greedy solution.
    """
    problem = _prepare(df, must_include, must_exclude)
    n = problem.matrix.shape[0]
    extra_k = k - len(problem.must_include)
    n_candidates = len(problem.candidates)
    if extra_k < 0 or extra_k > n_candidates:
        return TurfSearch(pd.DataFrame(columns=RESULT_COLUMNS), False, 0, 0.0)

    cand_matrix = problem.matrix[:, [problem.index[c] for c in problem.candidates]]
    base_reached, base_freq = _base(problem)
    total = math.comb(n_candidates, extra_k)

    if total > max_combinations:
        target = min(max_combinations, int(total * SAMPLE_SPACE_CAP))
        sampled = _sample_combinations(n_candidates, extra_k, target, seed)
        greedy = greedy_turf(df, k, problem.must_include, must_exclude)
        position = {name: i for i, name in enumerate(problem.candidates)}
        greedy_extra = tuple(
            sorted(position[name] for name in greedy.portfolio[len(problem.must_include):k])
        )
        if len(greedy_extra) == extra_k:
            sampled.add(greedy_extra)  # a set, so an already-sampled portfolio is not duplicated
        combos = np.array(sorted(sampled), dtype=np.intp).reshape(-1, extra_k)
        is_sampled = True
        coverage = round(target / total * 100, 1)
    else:
        combos = _all_combinations(n_candidates, extra_k)
        is_sampled = False
        coverage = 100.0

    counts, freqs = _evaluate(cand_matrix, base_reached, base_freq, combos)
    labels = [
        " + ".join(map(str, problem.must_include + [problem.candidates[i] for i in row]))
        for row in combos.tolist()
    ]
    frame = pd.DataFrame({
        "Combination": labels,
        "Reach (%)": np.round(counts / n * 100, 2),
        "Reach (Count)": counts,
        "Frequency": freqs,
        "_order": np.arange(len(labels)),
    })
    frame = (
        frame.sort_values(
            ["Reach (Count)", "Frequency", "_order"], ascending=[False, False, True]
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return TurfSearch(frame, is_sampled, total, coverage)


def individual_reach(df: pd.DataFrame, must_exclude: Sequence[str] = ()) -> pd.DataFrame:
    """Stand-alone reach of every item that is not excluded."""
    n = len(df)
    if n == 0:
        raise ValueError("The data contains no respondents.")
    excluded = set(must_exclude)
    counts = {c: int((df[c] > 0).sum()) for c in df.columns if c not in excluded}
    frame = pd.DataFrame({"Item": list(counts), "Reach (Count)": list(counts.values())})
    frame["Reach (%)"] = (frame["Reach (Count)"] / n * 100).round(2)
    return (
        frame[["Item", "Reach (%)", "Reach (Count)"]]
        .sort_values("Reach (Count)", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def greedy_turf(
    df: pd.DataFrame,
    max_k: int,
    must_include: Sequence[str] = (),
    must_exclude: Sequence[str] = (),
) -> GreedyResult:
    """Build a portfolio one item at a time, always adding the item with the most new reach.

    ``max_k`` is the total portfolio size including must-include items. Ties are broken by
    higher stand-alone frequency, then by column order. This is a fast heuristic: it is
    deterministic but not guaranteed to find the best possible portfolio.
    """
    problem = _prepare(df, must_include, must_exclude)
    matrix = problem.matrix
    n = matrix.shape[0]
    covered, _ = _base(problem)
    frequency = matrix.sum(axis=0)

    portfolio = list(problem.must_include)
    reach_curve = [_pct(covered.sum(), n)]
    gains_pct: list[float] = []
    gains_count: list[int] = []

    remaining = [problem.index[c] for c in problem.candidates]
    slots = max(0, max_k - len(portfolio))
    for _ in range(min(slots, len(remaining))):
        gains = (matrix[:, remaining] & ~covered[:, None]).sum(axis=0)
        pick = int(np.lexsort((-np.arange(len(remaining)), frequency[remaining], gains))[-1])
        column = remaining.pop(pick)
        covered = covered | matrix[:, column]
        portfolio.append(problem.items[column])
        reach_curve.append(_pct(covered.sum(), n))
        gains_pct.append(_pct(gains[pick], n))
        gains_count.append(int(gains[pick]))
    return GreedyResult(portfolio, reach_curve, gains_pct, gains_count)


def optimal_by_size(
    df: pd.DataFrame,
    max_k: int,
    must_include: Sequence[str] = (),
    must_exclude: Sequence[str] = (),
    max_combinations: int = MAX_COMBINATIONS,
) -> pd.DataFrame:
    """Best portfolio at each size 1..max_k, with the incremental reach of each extra item.

    Exact where the search space allows it, otherwise the better of (a) the greedy portfolio
    and (b) the previous size's best portfolio plus its best extra item. Option (b)
    guarantees reach never decreases as the size grows.
    """
    problem = _prepare(df, must_include, must_exclude)
    matrix = problem.matrix
    n = matrix.shape[0]
    mi = problem.must_include
    candidates = problem.candidates
    cand_matrix = matrix[:, [problem.index[c] for c in candidates]]
    base_reached, base_freq = _base(problem)
    greedy = greedy_turf(df, max_k, mi, must_exclude)

    def union_count(names: Sequence[str]) -> int:
        return int(matrix[:, [problem.index[x] for x in names]].any(axis=1).sum())

    rows = []
    prev_count = 0
    prev_items: list[str] | None = None
    for size in range(max(1, len(mi)), max_k + 1):
        extra_k = size - len(mi)
        if extra_k > len(candidates):
            break
        total = math.comb(len(candidates), extra_k)
        if total <= max_combinations:
            combos = _all_combinations(len(candidates), extra_k)
            counts, freqs = _evaluate(cand_matrix, base_reached, base_freq, combos)
            best = _best_index(counts, freqs)
            items = mi + [candidates[i] for i in combos[best]]
            count = int(counts[best])
            method = "Exact"
        else:
            items = greedy.portfolio[:size]
            count = union_count(items)
            if prev_items is not None:
                reached = matrix[:, [problem.index[x] for x in prev_items]].any(axis=1)
                gains = (cand_matrix & ~reached[:, None]).sum(axis=0)
                for position, name in enumerate(candidates):
                    if name in prev_items:
                        gains[position] = -1
                best_extra = int(np.argmax(gains))
                if prev_count + int(gains[best_extra]) > count:
                    items = [*prev_items, candidates[best_extra]]
                    count = prev_count + int(gains[best_extra])
            method = "Greedy approx."
        rows.append({
            "Size": size,
            "Combination": " + ".join(map(str, items)),
            "Reach (%)": _pct(count, n),
            "Reach (Count)": count,
            "Incremental Reach (%)": _pct(count - prev_count, n),
            "Method": method,
        })
        prev_count, prev_items = count, items
    return pd.DataFrame(
        rows,
        columns=["Size", "Combination", "Reach (%)", "Reach (Count)", "Incremental Reach (%)", "Method"],
    )


def diminishing_returns_size(
    sizes: Sequence[int], increments: Sequence[float], threshold: float = 2.0
) -> int | None:
    """Portfolio size after which the next item adds less than ``threshold`` points of reach.

    The first entry is never treated as the cut-off, so the recommendation is at least
    ``sizes[0]``. Returns ``None`` if the gain never falls below the threshold.
    """
    for i in range(1, len(increments)):
        if increments[i] < threshold:
            return int(sizes[i - 1])
    return None
