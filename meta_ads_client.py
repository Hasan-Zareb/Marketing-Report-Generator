"""
Meta Ads Manager API client.
Fetches ad-level spend data from Meta Graph API v25.0 for Facebook Web campaigns.

Two ad accounts are queried:
  - act_2484639905284235 (Alright TV - Micro drama App)
  - act_1505162154172822 (Alright TV- Ads)

Filters: campaign name CONTAIN "web", ad status IN [ACTIVE, PAUSED, ARCHIVED, DELETED].
Uses a wide date range (2 years back) since date_preset=lifetime is deprecated in v25.0.
"""

import json
import os
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

API_VERSION = "v25.0"
BASE_URL = f"https://graph.facebook.com/{API_VERSION}"

AD_ACCOUNTS = [
    {"id": "act_2484639905284235", "name": "Alright TV - Micro drama App"},
    {"id": "act_1505162154172822", "name": "Alright TV- Ads"},
]

FIELDS = ",".join([
    "date_start",
    "date_stop",
    "campaign_name",
    "campaign_id",
    "adset_name",
    "adset_id",
    "ad_name",
    "ad_id",
    "impressions",
    "spend",
    "inline_link_clicks",
])

FILTERS = (
    '['
    '{"field":"campaign.name","operator":"CONTAIN","value":"web"},'
    '{"field":"ad.effective_status","operator":"IN","value":["ACTIVE","PAUSED","ARCHIVED","DELETED"]}'
    ']'
)

# Timezone for insights so date_start/date_stop align with Ads Manager UI/export (India).
# Uses IANA timezone name; matches Meta timezone ID Asia/Kolkata (TZ_ASIA_KOLKATA).
INSIGHTS_TIMEZONE = "Asia/Kolkata"


def _get_access_token() -> Optional[str]:
    token = os.getenv("META_ACCESS_TOKEN", "").strip()
    if not token or token in ("PASTE_YOUR_TOKEN_HERE", "your_meta_token_here"):
        return None
    return token


def _default_time_range() -> dict:
    """Return a time_range covering 6 months back to today."""
    end = date.today()
    start = end - timedelta(days=180)
    return {"since": start.isoformat(), "until": end.isoformat()}


MAX_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]
TRANSIENT_CODES = {1, 2, 4, 17, 32, 190}


def _is_transient(error: dict) -> bool:
    """Return True if the Meta API error is likely transient and worth retrying."""
    return error.get("code", 0) in TRANSIENT_CODES or error.get("is_transient", False)


def _request_with_retry(method: str, url: str, account_name: str, **kwargs) -> dict:
    """Make an HTTP request with automatic retry on transient Meta API errors."""
    for attempt in range(MAX_RETRIES + 1):
        resp = requests.request(method, url, timeout=120, **kwargs)
        data = resp.json()
        if "error" not in data:
            return data
        if attempt < MAX_RETRIES and _is_transient(data["error"]):
            wait = RETRY_BACKOFF[attempt]
            print(f"  Meta API transient error on {account_name}, retrying in {wait}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
            continue
        raise RuntimeError(
            f"Meta API error ({account_name}): {data['error'].get('message', data['error'])}"
        )
    raise RuntimeError(f"Meta API failed after {MAX_RETRIES} retries ({account_name})")


def fetch_insights_sync(account_id: str, account_name: str, access_token: str) -> list[dict]:
    """Fetch all insights for an ad account using synchronous GET requests with pagination."""
    all_rows = []
    url = f"{BASE_URL}/{account_id}/insights"
    params = {
        "fields": FIELDS,
        "time_range": json.dumps(_default_time_range()),
        "level": "ad",
        "time_increment": 1,
        "filtering": FILTERS,
        "limit": 500,
        "timezone_name": INSIGHTS_TIMEZONE,
        "access_token": access_token,
    }

    while url:
        data = _request_with_retry("GET", url, account_name, params=params)
        rows = data.get("data", [])
        for row in rows:
            row["ad_account_name"] = account_name
            row["ad_account_id"] = account_id
        all_rows.extend(rows)
        next_url = data.get("paging", {}).get("next")
        url = next_url if next_url else None
        params = {}

    return all_rows


def fetch_insights_async(account_id: str, account_name: str, access_token: str) -> list[dict]:
    """Fetch insights using async jobs -- better for large accounts."""
    params = {
        "fields": FIELDS,
        "time_range": json.dumps(_default_time_range()),
        "level": "ad",
        "time_increment": 1,
        "filtering": FILTERS,
        "limit": 500,
        "timezone_name": INSIGHTS_TIMEZONE,
        "access_token": access_token,
    }

    job_response = _request_with_retry(
        "POST", f"{BASE_URL}/{account_id}/insights", account_name, params=params
    )
    report_run_id = job_response["report_run_id"]

    while True:
        status_response = _request_with_retry(
            "GET", f"{BASE_URL}/{report_run_id}", account_name,
            params={"access_token": access_token},
        )
        status = status_response.get("async_status", "unknown")
        if status in ("Job Complete", "Job Completed"):
            break
        elif status in ("Job Failed", "Job Skipped"):
            raise RuntimeError(f"Meta async job failed ({account_name}): {status}")
        time.sleep(5)

    all_rows = []
    url = f"{BASE_URL}/{report_run_id}/insights"
    fetch_params = {"access_token": access_token, "limit": 500}

    while url:
        data = _request_with_retry("GET", url, account_name, params=fetch_params)
        rows = data.get("data", [])
        for row in rows:
            row["ad_account_name"] = account_name
            row["ad_account_id"] = account_id
        all_rows.extend(rows)
        next_url = data.get("paging", {}).get("next")
        url = next_url if next_url else None
        fetch_params = {}

    return all_rows


def fetch_all_accounts(use_async: bool = True, access_token: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch data from both ad accounts, merge, clean, and return a DataFrame.

    Columns: Day, Campaign Name, Ad Set Name, Ad Name, Impressions, Amount Spent,
    Link Clicks, Ad Account, ad_account_id, campaign_id, adset_id, ad_id, date_stop.
    """
    if access_token is None:
        access_token = _get_access_token()
    if not access_token:
        raise ValueError("META_ACCESS_TOKEN is not set. Add it to .env or Streamlit Secrets.")

    all_rows = []
    for account in AD_ACCOUNTS:
        try:
            if use_async:
                rows = fetch_insights_async(account["id"], account["name"], access_token)
            else:
                rows = fetch_insights_sync(account["id"], account["name"], access_token)
        except RuntimeError:
            if not use_async:
                print(f"  Sync fetch failed for {account['name']}, falling back to async...")
                rows = fetch_insights_async(account["id"], account["name"], access_token)
            else:
                raise
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date_start"] = pd.to_datetime(df["date_start"])
    df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce").fillna(0).astype(int)
    df["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0).round(2)
    if "inline_link_clicks" in df.columns:
        df["inline_link_clicks"] = pd.to_numeric(df["inline_link_clicks"], errors="coerce").fillna(0).astype(int)
    else:
        df["inline_link_clicks"] = 0

    df = df.rename(columns={
        "date_start":         "Day",
        "campaign_name":      "Campaign Name",
        "adset_name":         "Ad Set Name",
        "ad_name":            "Ad Name",
        "impressions":        "Impressions",
        "spend":              "Amount Spent",
        "inline_link_clicks": "Link Clicks",
        "ad_account_name":    "Ad Account",
    })

    df = df.sort_values(
        by=["Day", "Campaign Name", "Ad Set Name", "Ad Name"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    return df


def filter_by_date(df: pd.DataFrame, since: str, until: str) -> pd.DataFrame:
    """Filter already-fetched DataFrame by a date range (no API call)."""
    if df.empty:
        return df
    mask = (df["Day"] >= pd.to_datetime(since)) & (df["Day"] <= pd.to_datetime(until))
    return df[mask].reset_index(drop=True)
