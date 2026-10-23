# 📊 TURF Analysis Automation Tool

A Streamlit app for **TURF (Total Unduplicated Reach and Frequency)** analysis. Given respondent-level
data, it finds the combination of items (products, features, messages, media touchpoints) that reaches
the largest number of *unique* respondents.

## Features

- **Data in**: upload a CSV or Excel file (multi-sheet workbooks supported) or use built-in simulated data.
- **Forgiving input**: accepts `0/1`, `True/False` and `Yes/No` columns; blanks count as "not selected";
  non-binary numbers are converted (`> 0` → 1) with a visible warning; ID/metadata columns are
  detected and unselected by default.
- **Constraints**: lock items into every portfolio (*must-include*) or remove them (*must-exclude*).
- **Five result views**, each downloadable as CSV:
  - **Individual Reach** – stand-alone reach of every item
  - **Optimal by Size** – best portfolio at each size 1…k with incremental reach and a diminishing-returns hint
  - **Top Combinations** – ranked portfolios of exactly *k* items, with reach and frequency
  - **Greedy Path** – items added one at a time by largest gain in new respondents
  - **Reach Curve** – cumulative reach and marginal gain per added item

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501.

### Docker

```bash
docker build -t turf .
docker run --rm -p 8501:8501 turf
```

The image runs as a non-root user and exposes a health check at `/_stcore/health`.

### Streamlit Community Cloud

Point a new app at this repository with `streamlit_app.py` as the entry point. No secrets are needed.

## Input data format

| Rows | Columns | Values |
| --- | --- | --- |
| One respondent per row | One item per column (header = item name) | `1` selected / `0` not selected |

- Remove (or deselect in the app) ID, demographic and timestamp columns.
- Up to 200 item columns can be analysed; portfolio sizes up to 12 are offered.
- Maximum upload size is 50 MB (`.streamlit/config.toml`).

## How the analysis works

- **Reach** = share of respondents who selected at least one item in the portfolio.
  **Frequency** = total selections across the portfolio's items.
- **Top Combinations** ranks by reach, then frequency. If the number of possible combinations is
  **≤ 15,000** every one is evaluated (exact). Otherwise a reproducible random sample (fixed seed,
  at most 75% of the space) is evaluated and the greedy portfolio is added to it. The app labels this
  as sampled and shows the coverage; the best result shown is then **not guaranteed** to be the true optimum.
- **Optimal by Size** is exact wherever the search space allows; other sizes are marked
  *Greedy approx.* and are guaranteed never to reach less than the previous size.
- **Greedy** is a fast, deterministic heuristic (ties broken by higher frequency, then column order).
  It can miss a better combination found by the full search.
- The *diminishing returns* hint flags the size after which an extra item adds less than 2 points of reach.

## Project layout

```
streamlit_app.py     Streamlit UI: input, caching, charts, downloads
turf_core.py         Pure TURF maths and data cleaning (no Streamlit dependency)
tests/               Unit tests (maths, cleaning, file reading) + headless end-to-end UI tests
.streamlit/          Server settings (upload limit, XSRF protection, no telemetry)
Dockerfile           Production container
.github/workflows/   CI: lint + tests on Python 3.11–3.13
```

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The tests compare the optimised search against brute force, check every cleaning rule, and drive the
real app headlessly (including uploads on Streamlit versions whose test harness supports them).

## Security and privacy

- Uploaded files are processed in memory and never written to disk by the app. Streamlit caches
  derived data for up to one hour in a bounded cache; restarting the server clears it.
- XSRF protection is enabled and usage telemetry is disabled.
- The app has no authentication. If it handles confidential respondent data, deploy it behind your
  organisation's authentication (reverse proxy, SSO, or a private deployment).

## License

Apache 2.0 — see [LICENSE](LICENSE).
