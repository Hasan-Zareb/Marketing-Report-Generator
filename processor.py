"""Core logic: Xlookup (abbreviations), pivot, Daily/Weekly output."""

import re
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
    "Adgroup name": "adgroup_name",  # Used for abbreviation lookup (not in CANONICAL_COLUMNS)
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

# Hardcoded abbreviation -> Show name. Ad name column is searched; first match maps to the show.
# Keys are uppercase for matching; values are display show names.
ABBREVIATION_TO_SHOW = {
    "BKC": "boss ka comeback",
    "CDS": "ceo's dirty secret",
    "CKW": "ceo ki wapsi",
    "COMB": "crushing on my bodygaurd",
    "CYN": "chehra ya nakaab",
    "DSDT": "dil se dil tak",
    "FRTA": "from rivals to allies",
    "FWL": "Forever was a lie",
    "HAK": "humari adhoori kahani",
    "IO": "independent organ",
    "IPUP": "IAS pyaar under pressure",
    "LGRO": "love gone revenge on",
    "LOD": "love or divorce",
    "LR": "love racer",
    "LAAPATA LADY": "laapata lady",
    "LOCKED IN": "locked in",
    "MAFS": "married at first sight",
    "MBDLF": "my billion dollar love fund",
    "MBHC": "maa beta aur hidden ceo",
    "MBMD": "maa beta aur millionaire doctor",
    "MBVD": "maa beta aur vicky donor",
    "MHYW": "mr huo your wife",
    "MJ": "Maya Jaal",
    "MMHN": "married my husbands nemesis",
    "MWP": "mohalle wala pyaar",
    "OFNB": "one fateful night with my boss",
    "OLL": "one last love",
    "OLTOOS": "our love to ocean of stars",
    "PKPT": "pyaar ka price tag",
    "QPSB": "qatil pati se badla",
    "QWC": "qismat wala connection",
    "SBM": "shaadi by mistake",
    "SKG": "suhaag ka gamble",
    "SMC": "saving mr ceo",
    "STS": "shaadi to soulmates",
    "SWS": "saazish wali shaadi",
    "TCC": "the comeback ceo",
    "TDPP": "the devil prince's pursuit",
    "TVW": "laapata lady",
    "TROPHY WIFE": "Trophy Wife",
    "WKMW": "who killed my wife",
    "WAARIS": "Waaris",
    "ASHVRAT": "ashvrat",
    "BBB": "band baaja billionaire",
}
UNMAPPED_LABEL = "no"  # Match Excel report for unmapped adgroups


def _find_show_name_by_abbreviation(text: str, abbrev_to_show: Optional[dict[str, str]] = None) -> Optional[str]:
    """
    Search text (e.g. adgroup name or ad name) for hardcoded abbreviations.
    Returns Show name if any abbreviation is found; uses longest matches first.
    """
    abbrev_to_show = abbrev_to_show or ABBREVIATION_TO_SHOW
    if not text or not abbrev_to_show:
        return None
    text_upper = str(text).upper()
    # Sort by length descending so "MHYW" matches before "MHY"
    for abbr in sorted(abbrev_to_show.keys(), key=len, reverse=True):
        # Word-boundary style: abbreviation as a distinct token (surrounded by non-alphanumeric)
        pattern = r"(?<![A-Za-z0-9])" + re.escape(abbr) + r"(?![A-Za-z0-9])"
        if re.search(pattern, text_upper):
            return abbrev_to_show[abbr]
    return None


def xlookup_by_abbreviations(input_df: pd.DataFrame) -> pd.DataFrame:
    """Map Show Name by searching the Ad name column (creative_network) for hardcoded abbreviations."""
    input_df = input_df.copy()

    if "creative_network" not in input_df.columns:
        input_df["Show Name"] = UNMAPPED_LABEL
        return input_df

    def resolve(row):
        val = row.get("creative_network")
        return _find_show_name_by_abbreviation(str(val) if pd.notna(val) else "", ABBREVIATION_TO_SHOW)

    show_names = input_df.apply(resolve, axis=1)
    input_df["Show Name"] = show_names.fillna(UNMAPPED_LABEL)

    # Insert Show Name after creative_network if present
    cols = list(input_df.columns)
    if "creative_network" in cols:
        cn_idx = cols.index("creative_network")
        if "Show Name" in cols:
            cols.remove("Show Name")
        cols.insert(cn_idx + 1, "Show Name")
        input_df = input_df[cols]
    return input_df


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
    # Normalize Show Name to lowercase for consistent grouping (handles case variations)
    # Then convert to title case for display, preserving acronyms like "IAS"
    # Keep "no" and "(Unmapped)" as separate, distinct entries
    df = df.copy()
    show_names = df["Show Name"].astype(str)
    
    # Convert to lowercase for grouping (to merge duplicates)
    show_names_lower = show_names.str.lower().str.strip()
    
    # Preserve special cases: "no" and "(Unmapped)" should remain separate
    # Use a mapping to preserve these exactly
    normalized = show_names_lower.str.title()
    
    # Fix special cases
    normalized = normalized.str.replace(r'^Ias\s+', 'IAS ', regex=True)  # Preserve IAS acronym
    normalized = normalized.where(show_names_lower != 'no', 'No')  # Keep "no" (unmapped) as "No" - match Excel
    
    df["Show Name"] = normalized
    
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
    daily_date: Optional[pd.Timestamp] = None,
    weekly_start_date: Optional[pd.Timestamp] = None,
) -> tuple:
    """
    Run Xlookup (abbreviations) -> pivot -> Daily/Weekly outputs.

    Args:
        input_df: Input CSV DataFrame
        daily_date: Date for daily report (defaults to yesterday if not provided)
        weekly_start_date: Start date for weekly report (defaults to 7 days ago if not provided).
                          Weekly report includes 7 days from this date (inclusive).

    Returns:
        (daily_output, weekly_output) DataFrames with columns Show Name, #Free Trials, #Subscriptions,
        Ad Spend, Cost of Free Trial, CAC.
    """
    df = normalize_input_columns(input_df)
    xl = xlookup_by_abbreviations(df)

    for col in ("free_trial_be_events", "revenue_3ea7d4b1_events", "cost"):
        xl[col] = pd.to_numeric(xl[col], errors="coerce").fillna(0)

    xl["day"] = _parse_day(xl["day"])
    xl = xl.dropna(subset=["day"])

    if xl.empty:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty.copy(), empty.copy()

    # Daily: use provided date or default to yesterday
    if daily_date is None:
        daily_date = _yesterday()
    else:
        daily_date = pd.Timestamp(daily_date).normalize()
    
    daily_filtered = _filter_by_dates(xl, [daily_date])
    daily_pivot = _pivot(daily_filtered)
    daily_period = _format_date(daily_date)
    daily_out = _build_output(daily_pivot, daily_period)

    # Weekly: use provided start date or default to 7 days ago
    if weekly_start_date is None:
        week_dates = _past_7_days_ending_yesterday()
    else:
        weekly_start_date = pd.Timestamp(weekly_start_date).normalize()
        # Generate 7 days from start date (inclusive)
        week_dates = [weekly_start_date + pd.Timedelta(days=i) for i in range(7)]
    
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
    # Adgroup name is optional for new format (improves abbreviation matching)
    new_required = {k: v for k, v in NEW_TO_CANONICAL.items() if v != "adgroup_name"}
    if all(c in df.columns for c in CANONICAL_COLUMNS):
        return []
    if all(k in df.columns for k in new_required):
        return []
    missing_c = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    missing_n = [k for k in new_required if k not in df.columns]
    return missing_n if len(missing_n) <= len(missing_c) else missing_c


def export_csv(df: pd.DataFrame) -> str:
    """Export DataFrame to CSV string (plain numbers, minimal formatting)."""
    return df.to_csv(index=False)
