"""Core logic: Xlookup, pivot, Daily/Weekly output."""

from pathlib import Path
from typing import Optional

import pandas as pd

# Canonical names used in the pipeline (old input format)
CANONICAL_COLUMNS = [
    "creative_network",
    "day",
    "free_trial_be_events",
    "revenue_3ea7d4b1_events",
    "cost",
]

# New input format -> canonical
NEW_TO_CANONICAL = {
    "Ad name": "creative_network",
    "Day (date)": "day",
    "FREE_TRIAL_BE": "free_trial_be_events",
    "REVENUE": "revenue_3ea7d4b1_events",
    "Ad spend": "cost",
}

REQUIRED_COLUMNS = CANONICAL_COLUMNS

OUTPUT_COLUMNS = [
    "Show Name",
    "#Free Trials",
    "#Subscriptions",
    "Ad Spend",
    "Cost of Free Trial",
    "CAC",
]


def _resolve_reference_path() -> Path:
    """Path to bundled Show Name Reference.csv (same folder as this module)."""
    return Path(__file__).resolve().parent / "Show Name Reference.csv"


def load_reference(path: Optional[Path] = None) -> pd.DataFrame:
    """Load Show Name Reference CSV. Trims creative_network for matching."""
    p = path or _resolve_reference_path()
    df = pd.read_csv(p, encoding="latin-1")
    df["creative_network"] = df["creative_network"].astype(str).str.strip()
    return df


def xlookup(input_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join on creative_network; add Show Name after creative_network. Unmapped -> (Unmapped)."""
    input_df = input_df.copy()
    input_df["creative_network"] = input_df["creative_network"].astype(str).str.strip()

    # Handle both old format ("show") and new format ("Show name")
    show_col = "Show name" if "Show name" in reference_df.columns else "show"
    ref = reference_df[["creative_network", show_col]].drop_duplicates(subset="creative_network")
    merged = input_df.merge(
        ref.rename(columns={show_col: "Show Name"}),
        on="creative_network",
        how="left",
    )
    merged["Show Name"] = merged["Show Name"].fillna("(Unmapped)")

    cols = list(merged.columns)
    cn_idx = cols.index("creative_network")
    sn_idx = cols.index("Show Name")
    if sn_idx != cn_idx + 1:
        cols.remove("Show Name")
        cols.insert(cn_idx + 1, "Show Name")
        merged = merged[cols]
    return merged


def _parse_day(ser: pd.Series) -> pd.Series:
    """Parse day as datetime; invalid/missing -> NaT."""
    return pd.to_datetime(ser, errors="coerce")


def _latest_day(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Latest valid day in data."""
    dates = df["day"].dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max())


def _latest_complete_week_dates(df: pd.DataFrame) -> Optional[list]:
    """Dates of latest complete ISO week (Mon–Sun) present in data; else week containing latest day."""
    dates = df["day"].dropna()
    if dates.empty:
        return None
    dates = pd.to_datetime(dates)
    data_dates = set(dates.dt.normalize())
    latest = pd.Timestamp(dates.max()).normalize()
    y, w, _ = latest.isocalendar()
    fallback_mon = pd.Timestamp.fromisocalendar(int(y), int(w), 1)
    fallback_week = [fallback_mon + pd.Timedelta(days=i) for i in range(7)]

    mon = fallback_mon
    for _ in range(53):
        week_dates = [mon + pd.Timedelta(days=i) for i in range(7)]
        if all(d in data_dates for d in week_dates):
            return week_dates
        mon -= pd.Timedelta(days=7)
    return fallback_week


def _filter_by_dates(df: pd.DataFrame, keep_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Restrict to rows whose day is in keep_dates."""
    keep_set = set(pd.Timestamp(x).date() for x in keep_dates)
    return df[df["day"].dt.date.isin(keep_set)]


def _pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Group by Show Name; sum free_trial_be_events, revenue_3ea7d4b1_events, cost. No Grand Total."""
    agg = (
        df.groupby("Show Name", as_index=False)
        .agg(
            free_trial_be_events=("free_trial_be_events", "sum"),
            revenue_3ea7d4b1_events=("revenue_3ea7d4b1_events", "sum"),
            cost=("cost", "sum"),
        )
    )
    return agg


def _build_output(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """Build output table: Show Name, #Free Trials, #Subscriptions, Ad Spend, Cost of Free Trial, CAC."""
    out = pivot_df.rename(columns={
        "free_trial_be_events": "#Free Trials",
        "revenue_3ea7d4b1_events": "#Subscriptions",
        "cost": "Ad Spend",
    })[["Show Name", "#Free Trials", "#Subscriptions", "Ad Spend"]]

    trials = out["#Free Trials"]
    subs = out["#Subscriptions"]
    spend = out["Ad Spend"]

    cost_of_trial = spend / trials
    cost_of_trial = cost_of_trial.where(trials > 0, "-")
    cac = spend / subs
    cac = cac.where(subs > 0, "-")

    out["Cost of Free Trial"] = cost_of_trial
    out["CAC"] = cac

    # Round all numeric columns to 1 decimal place
    numeric_cols = ["#Free Trials", "#Subscriptions", "Ad Spend", "Cost of Free Trial", "CAC"]
    for col in numeric_cols:
        if col in out.columns:
            # Convert to float and round numeric values, keep "-" as strings
            out[col] = out[col].apply(
                lambda x: round(float(x), 1) if isinstance(x, (int, float)) and not pd.isna(x) else x
            )

    return out[OUTPUT_COLUMNS]


def process(
    input_df: pd.DataFrame,
    reference_path: Optional[Path] = None,
) -> tuple:
    """
    Run Xlookup -> pivot -> Daily/Weekly outputs.

    Returns:
        (daily_output, weekly_output) DataFrames with columns Show Name, #Free Trials, #Subscriptions,
        Ad Spend, Cost of Free Trial, CAC.
    """
    df = normalize_input_columns(input_df)
    ref = load_reference(reference_path)
    xl = xlookup(df, ref)

    for col in ("free_trial_be_events", "revenue_3ea7d4b1_events", "cost"):
        xl[col] = pd.to_numeric(xl[col], errors="coerce").fillna(0)

    xl["day"] = _parse_day(xl["day"])
    xl = xl.dropna(subset=["day"])

    if xl.empty:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty.copy(), empty.copy()

    # Daily: latest day
    latest = _latest_day(xl)
    if latest is None:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty.copy(), empty.copy()

    daily_filtered = _filter_by_dates(xl, [latest])
    daily_pivot = _pivot(daily_filtered)
    daily_out = _build_output(daily_pivot)

    # Weekly: latest complete week (or partial)
    week_dates = _latest_complete_week_dates(xl)
    if not week_dates:
        weekly_out = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        weekly_filtered = _filter_by_dates(xl, week_dates)
        weekly_pivot = _pivot(weekly_filtered)
        weekly_out = _build_output(weekly_pivot)

    return daily_out, weekly_out


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If input uses new-format column names, rename to canonical. Else return as-is."""
    if all(k in df.columns for k in NEW_TO_CANONICAL):
        return df.rename(columns=NEW_TO_CANONICAL)
    return df


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return list of missing required column names. Empty if all present (old or new format)."""
    if all(c in df.columns for c in CANONICAL_COLUMNS):
        return []
    if all(k in df.columns for k in NEW_TO_CANONICAL):
        return []
    missing_c = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    missing_n = [k for k in NEW_TO_CANONICAL if k not in df.columns]
    return missing_n if len(missing_n) <= len(missing_c) else missing_c


def export_csv(df: pd.DataFrame) -> str:
    """Export DataFrame to CSV string (plain numbers, minimal formatting)."""
    return df.to_csv(index=False)
