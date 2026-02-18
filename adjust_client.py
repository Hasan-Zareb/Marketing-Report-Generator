"""
Adjust Report Service API client.
Fetches CSV report from the same data as the **Marketing Performance** report in Datascape.

Endpoint: https://automate.adjust.com/reports-service/csv_report
Docs: https://dev.adjust.com/en/api/rs-api and https://dev.adjust.com/en/api/rs-api/csv/

Report we match: Marketing Performance (Datascape)
  URL (filter params): channel_id__in=partner_34,partner_254; dimensions=channel,ad_account_id,
  campaign_network,adgroup_network,creative_network,day; metrics=installs,sign_up_events,
  free_trial_be_events,revenue_3ea7d4b1_events,revenue_3ea7d4b1_revenue,cost,roas_iap;
  cohort_maturity=immature; attribution_source=first; ad_spend_mode=network.
  We request only these two channels so the API returns the same rows as the CSV export of that report.
"""

from datetime import date
from typing import Optional

import pandas as pd
import requests

ADJUST_CSV_URL = "https://automate.adjust.com/reports-service/csv_report"

# Marketing Performance report: only partners 34 and 254 (same as report URL channel_id__in)
MARKETING_PERFORMANCE_CHANNEL_IDS = "partner_34,partner_254"

# Adjust dimension/metric slugs that map to our input format
# Dimensions: partner_name=Channel, campaign, adgroup_network=Adgroup name, creative_network=Ad name, day
# Metrics: installs, network_cost, + custom events (FREE_TRIAL_BE, REVENUE from Events API)
DEFAULT_DIMENSIONS = "partner_name,campaign,adgroup_network,creative_network,day"
DEFAULT_METRICS = "installs,network_cost,free_trial_be_events,revenue_3ea7d4b1_events"

# Column mapping: Adjust API response (slug or readable name) -> our input format
ADJUST_TO_INPUT = {
    "partner_name": "Channel",
    "partner": "Channel",
    "campaign": "Campaign name",
    "campaign_network": "Campaign name",
    "adgroup_network": "Adgroup name",
    "adgroup": "Adgroup name",
    "creative_network": "Ad name",
    "creative": "Ad name",
    "day": "Day (date)",
    "installs": "Installs",
    "network_cost": "Ad spend",
    "cost": "Ad spend",
    "network_installs": "Installs",
    "sign_up": "SIGN_UP",
    "free_trial_be": "FREE_TRIAL_BE",
    "free_trial_be_events": "FREE_TRIAL_BE",
    "revenue": "REVENUE",
    "revenue_3ea7d4b1_events": "REVENUE",
}


def fetch_adjust_report(
    api_token: str,
    date_start: date,
    date_end: date,
    app_token: Optional[str] = None,
    dimensions: str = DEFAULT_DIMENSIONS,
    metrics: str = DEFAULT_METRICS,
    extra_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Fetch CSV report from Adjust Report Service API.

    Args:
        api_token: Adjust API token (Bearer) - required
        date_start: Start date
        date_end: End date
        app_token: Optional app token(s), comma-separated. Omit to use all apps accessible via API token.
        dimensions: Comma-separated dimension slugs
        metrics: Comma-separated metric slugs (add custom event slugs for SIGN_UP, FREE_TRIAL_BE, REVENUE)
        extra_params: Optional extra query params (e.g. ad_spend_mode, readable_names)

    Returns:
        DataFrame with columns mapped to our input format (Channel, Ad account ID, Campaign name,
        Adgroup name, Ad name, Day (date), Installs, SIGN_UP, FREE_TRIAL_BE, REVENUE, Ad spend)
    """
    date_period = f"{date_start.isoformat()}:{date_end.isoformat()}"
    params = {
        "date_period": date_period,
        "dimensions": dimensions,
        "metrics": metrics,
        "ad_spend_mode": "network",
        "readable_names": "false",  # Use slugs for reliable column mapping
        # Marketing Performance report alignment (same as Datascape report URL)
        "channel_id__in": MARKETING_PERFORMANCE_CHANNEL_IDS,
        "cohort_maturity": "immature",
        "attribution_source": "first",
    }
    if app_token and app_token.strip():
        params["app_token__in"] = app_token.strip()
    if extra_params:
        params.update(extra_params)

    headers = {"Authorization": f"Bearer {api_token}"}
    resp = requests.get(ADJUST_CSV_URL, params=params, headers=headers, timeout=120)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        detail = resp.text[:300].strip() if resp.text else ""
        raise requests.HTTPError(
            f"Adjust API {e}: {detail}" if detail else str(e),
            response=resp,
        ) from e

    df = pd.read_csv(
        __import__("io").BytesIO(resp.content),
        encoding="utf-8-sig",
    )
    # Strip BOM from column names
    df.columns = [c.lstrip("\ufeff") if isinstance(c, str) else c for c in df.columns]

    # Map Adjust columns to our expected input format
    result = _map_to_input_format(df)
    return result


def _map_to_input_format(df: pd.DataFrame) -> pd.DataFrame:
    """Map Adjust API columns to our input CSV format."""
    rename = {}
    for adj_col in df.columns:
        adj_str = str(adj_col)
        adj_lower = adj_str.lower().replace(" ", "_").replace("-", "_")
        for slug, our_col in ADJUST_TO_INPUT.items():
            if slug == adj_lower or slug in adj_lower or adj_str == slug:
                rename[adj_col] = our_col
                break

    out = df.rename(columns={c: rename[c] for c in rename if c in df.columns})

    # Add missing required columns with placeholders
    metric_defaults = {"SIGN_UP": 0, "FREE_TRIAL_BE": 0, "REVENUE": 0, "Installs": 0, "Ad spend": 0}
    for col, default in metric_defaults.items():
        if col not in out.columns:
            out[col] = default
    if "Ad group name" in out.columns and "Adgroup name" not in out.columns:
        out["Adgroup name"] = out["Ad group name"]  # Handle space in "Ad group"
    if "Ad name" not in out.columns and "Creative network" in out.columns:
        out["Ad name"] = out["Creative network"]

    return out
