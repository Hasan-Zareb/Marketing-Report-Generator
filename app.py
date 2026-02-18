"""Streamlit UI: fetch from Adjust API and generate Daily/Weekly reports."""

import io
import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from adjust_client import fetch_adjust_report
from processor import (
    CANONICAL_COLUMNS,
    NEW_TO_CANONICAL,
    export_csv,
    process,
    validate_columns,
)

PLACEHOLDER_TOKENS = {"", "your_api_token_here"}


def _get_api_token():
    """Load API token from .env. Return None if invalid."""
    api = os.getenv("ADJUST_API_TOKEN", "").strip()
    if not api or api in PLACEHOLDER_TOKENS:
        return None
    return api


def main() -> None:
    st.set_page_config(page_title="Marketing Report Generator", layout="centered")
    st.title("Marketing Report Generator")

    api_token = _get_api_token()
    if not api_token:
        st.error(
            "Add your Adjust API token to `.env`: `ADJUST_API_TOKEN=your_token`. "
            "See `.env.example` for format."
        )
        st.stop()

    # Initialize session state
    if "daily_df" not in st.session_state:
        st.session_state.daily_df = None
    if "weekly_df" not in st.session_state:
        st.session_state.weekly_df = None

    # Step 1: Date selection (before Refresh)
    st.subheader("Report Dates")
    col1, col2 = st.columns(2)
    with col1:
        daily_date = st.date_input(
            "Daily Report Date",
            value=date.today() - timedelta(days=1),
            help="Date for the daily report",
            key="daily_date",
        )
    with col2:
        weekly_start_date = st.date_input(
            "Weekly Report Start Date",
            value=date.today() - timedelta(days=7),
            help="Start of the 7-day weekly report",
            key="weekly_start_date",
        )
    weekly_end_date = weekly_start_date + timedelta(days=6)
    st.caption(
        f"Weekly report: {weekly_start_date.strftime('%b %d, %Y')} – {weekly_end_date.strftime('%b %d, %Y')}"
    )

    # Step 2: Single Refresh button
    if st.button("Refresh – Fetch from Adjust & Generate Reports", type="primary", use_container_width=True):
        date_start = min(daily_date, weekly_start_date)
        date_end = max(daily_date, weekly_end_date)

        with st.spinner("Fetching data from Adjust API…"):
            try:
                df = fetch_adjust_report(
                    api_token=api_token,
                    date_start=date_start,
                    date_end=date_end,
                )
            except Exception as e:
                st.error(f"Adjust API error: {e}")
                if "your_api_token_here" in str(e):
                    st.info("Replace the placeholder in `.env` with your real Adjust API token.")
                st.stop()

        missing = validate_columns(df)
        if missing:
            st.error(
                f"Adjust returned data missing columns: {', '.join(missing)}. "
                "Check adjust_client.py for dimension/metric mapping."
            )
            st.stop()

        if df.empty:
            st.warning("Adjust returned no data for this date range.")
            st.stop()

        st.success(f"Fetched {len(df)} rows from Adjust.")

        with st.spinner("Generating Daily and Weekly reports…"):
            try:
                daily, weekly = process(
                    input_df=df,
                    daily_date=pd.Timestamp(daily_date),
                    weekly_start_date=pd.Timestamp(weekly_start_date),
                )
            except Exception as e:
                st.error(f"Processing failed: {e}")
                raise

        st.session_state.daily_df = daily
        st.session_state.weekly_df = weekly
        st.success("Reports ready.")

    # Show reports
    if st.session_state.daily_df is not None and st.session_state.weekly_df is not None:
        _show_tables_with_filters()

    # Optional: CSV upload fallback
    with st.expander("Or upload CSV instead", expanded=False):
        uploaded = st.file_uploader(
            "Upload input CSV",
            type=["csv"],
            key="csv_upload",
        )
        if uploaded is not None:
            try:
                raw = uploaded.read()
                df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
            except Exception:
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
                    df = None
            if df is not None:
                df.columns = [c.lstrip("\ufeff") if isinstance(c, str) else c for c in df.columns]
                missing = validate_columns(df)
                if not missing and not df.empty:
                    try:
                        daily, weekly = process(
                            input_df=df,
                            daily_date=pd.Timestamp(daily_date),
                            weekly_start_date=pd.Timestamp(weekly_start_date),
                        )
                        st.session_state.daily_df = daily
                        st.session_state.weekly_df = weekly
                        st.success("Reports generated from uploaded CSV.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
                elif missing:
                    st.error(f"Missing columns: {', '.join(missing)}")


def _apply_column_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df
    for col, allowed in filters.items():
        if col not in out.columns or allowed is None or len(allowed) == 0:
            continue
        out = out[out[col].astype(str).isin(allowed)]
    return out


def _safe_sort(df: pd.DataFrame, by: str, ascending: bool) -> pd.DataFrame:
    if by not in df.columns or df.empty:
        return df
    s = df[by]
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().any():
        tmp = "__sort_key__"
        out = df.assign(**{tmp: numeric}).sort_values(tmp, ascending=ascending, na_position="last")
        return out.drop(columns=[tmp])
    return df.sort_values(by=by, ascending=ascending, na_position="last")


def _table_with_filters(
    df: pd.DataFrame,
    title: str,
    export_filename: str,
    key_prefix: str,
) -> None:
    st.subheader(title)
    filters = {}
    with st.expander("Column filters", expanded=False):
        st.caption("All selected by default. Remove from selection to filter.")
        cols = st.columns(min(len(df.columns), 3))
        for i, col in enumerate(df.columns):
            with cols[i % len(cols)]:
                options = sorted(df[col].astype(str).unique().tolist())
                selected = st.multiselect(col, options=options, default=options, key=f"{key_prefix}_filter_{col}")
                filters[col] = set(selected) if selected else None
    filtered = _apply_column_filters(df, {k: v for k, v in filters.items() if v is not None})
    sc1, sc2 = st.columns(2)
    with sc1:
        sort_col = st.selectbox("Sort by", options=df.columns.tolist(), key=f"{key_prefix}_sort_col")
    with sc2:
        sort_asc = st.radio("Order", ["Ascending", "Descending"], horizontal=True, key=f"{key_prefix}_sort_order")
    filtered = _safe_sort(filtered, sort_col, sort_asc == "Ascending")
    st.download_button(
        f"Export {export_filename}",
        data=filtered.to_csv(index=False),
        file_name=export_filename,
        mime="text/csv",
        key=f"{key_prefix}_export",
    )
    st.dataframe(filtered, use_container_width=True, height=400)


def _show_tables_with_filters() -> None:
    st.divider()
    _table_with_filters(
        st.session_state.daily_df,
        "Daily Output",
        "Daily Output.csv",
        "daily",
    )
    st.divider()
    _table_with_filters(
        st.session_state.weekly_df,
        "Weekly Output",
        "Weekly Output.csv",
        "weekly",
    )


if __name__ == "__main__":
    main()
