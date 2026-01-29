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
    "Report Period",
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


def _yesterday() -> pd.Timestamp:
    """Return yesterday's date (today - 1 day)."""
    return pd.Timestamp.now().normalize() - pd.Timedelta(days=1)


def _past_7_days_ending_yesterday() -> list:
    """Return list of dates for the past 7 days ending yesterday (today - 7 through today - 1)."""
    yesterday = _yesterday()
    return [yesterday - pd.Timedelta(days=i) for i in range(6, -1, -1)]


def _format_date(date: pd.Timestamp) -> str:
    """Format date as 'Jan 29, 2026'."""
    return date.strftime("%b %d, %Y")


def _format_date_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Format date range as 'Jan 23 - Jan 29, 2026'."""
    if start.year == end.year:
        if start.month == end.month:
            return f"{start.strftime('%b %d')} - {end.strftime('%b %d, %Y')}"
        return f"{start.strftime('%b %d')} - {end.strftime('%b %d, %Y')}"
    return f"{start.strftime('%b %d, %Y')} - {end.strftime('%b %d, %Y')}"


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


def _build_output(pivot_df: pd.DataFrame, report_period: str) -> pd.DataFrame:
    """Build output table: Report Period, Show Name, #Free Trials, #Subscriptions, Ad Spend, Cost of Free Trial, CAC."""
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

    # Add Report Period as first column (same value for all rows)
    out.insert(0, "Report Period", report_period)

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

    # Daily: yesterday (today - 1)
    yesterday = _yesterday()
    daily_filtered = _filter_by_dates(xl, [yesterday])
    daily_pivot = _pivot(daily_filtered)
    daily_period = _format_date(yesterday)
    daily_out = _build_output(daily_pivot, daily_period)

    # Weekly: past 7 days ending yesterday (today - 7 through today - 1)
    week_dates = _past_7_days_ending_yesterday()
    weekly_filtered = _filter_by_dates(xl, week_dates)
    weekly_pivot = _pivot(weekly_filtered)
    weekly_period = _format_date_range(week_dates[0], week_dates[-1])
    weekly_out = _build_output(weekly_pivot, weekly_period)

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
