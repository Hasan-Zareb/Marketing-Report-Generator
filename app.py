"""Streamlit UI: upload input CSV, generate Daily/Weekly reports, download."""

import io

import pandas as pd
import streamlit as st

from processor import (
    REQUIRED_COLUMNS,
    _resolve_reference_path,
    export_csv,
    process,
    validate_columns,
)


def _ensure_reference() -> bool:
    """Validate bundled Show Name Reference.csv exists. Return True if ok."""
    p = _resolve_reference_path()
    if not p.exists():
        st.error(f"Show Name Reference.csv not found at {p}. Cannot run.")
        return False
    return True


def main() -> None:
    st.set_page_config(page_title="Marketing Report Generator", layout="centered")
    st.title("Marketing Report Generator")

    if not _ensure_reference():
        st.stop()

    # Initialize session state for storing generated reports
    if "daily_csv" not in st.session_state:
        st.session_state.daily_csv = None
    if "weekly_csv" not in st.session_state:
        st.session_state.weekly_csv = None
    if "daily_df" not in st.session_state:
        st.session_state.daily_df = None
    if "weekly_df" not in st.session_state:
        st.session_state.weekly_df = None
    if "last_file_id" not in st.session_state:
        st.session_state.last_file_id = None

    uploaded = st.file_uploader(
        "Upload input CSV (e.g. Input File 1.csv)",
        type=["csv"],
        help="Must include: creative_network, day, free_trial_be_events, revenue_3ea7d4b1_events, cost",
    )

    # Clear reports if a new file is uploaded
    if uploaded is not None:
        # Use file name and size as identifier to detect new uploads
        current_file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.last_file_id != current_file_id:
            st.session_state.daily_csv = None
            st.session_state.weekly_csv = None
            st.session_state.daily_df = None
            st.session_state.weekly_df = None
            st.session_state.last_file_id = current_file_id
    else:
        st.info("Upload an input CSV to continue.")
        # Show existing reports if available
        if st.session_state.daily_df is not None and st.session_state.weekly_df is not None:
            st.info("Previous reports are available below. Upload a new file to generate new reports.")
            _show_tables_with_filters()
        st.stop()

    try:
        raw = uploaded.read()
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
        except Exception as e2:
            st.error(f"Could not read CSV with latin-1: {e2}")
            st.stop()
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    missing = validate_columns(df)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}. Need: {', '.join(REQUIRED_COLUMNS)}.")
        st.stop()

    if df.empty:
        st.warning("Uploaded file is empty.")
        st.stop()

    if st.button("Generate reports"):
        with st.spinner("Generating Daily and Weekly outputs…"):
            try:
                daily, weekly = process(df)
            except Exception as e:
                st.error(f"Processing failed: {e}")
                raise

        st.success("Reports ready. View and export below.")

        # Store in session state (DataFrames + CSV)
        st.session_state.daily_df = daily
        st.session_state.weekly_df = weekly
        st.session_state.daily_csv = export_csv(daily)
        st.session_state.weekly_csv = export_csv(weekly)

    # Show tables with filters and export if reports are available
    if st.session_state.daily_df is not None and st.session_state.weekly_df is not None:
        _show_tables_with_filters()


def _apply_column_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply per-column filters (AND across columns). filters maps column -> set of allowed values."""
    out = df
    for col, allowed in filters.items():
        if col not in out.columns or allowed is None or len(allowed) == 0:
            continue
        out = out[out[col].astype(str).isin(allowed)]
    return out


def _safe_sort(df: pd.DataFrame, by: str, ascending: bool) -> pd.DataFrame:
    """Sort df by column; use numeric sort when possible, else lexicographic."""
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
    """Render a table with per-column filters (all selected by default; click to remove), sort, and export."""
    st.subheader(title)

    # Column filters: all selected by default; click to remove from selection and filter
    filters = {}
    with st.expander("Column filters", expanded=False):
        st.caption("All entries are selected by default. Click to remove from selection and filter.")
        cols = st.columns(min(len(df.columns), 3))
        for i, col in enumerate(df.columns):
            with cols[i % len(cols)]:
                options = sorted(df[col].astype(str).unique().tolist())
                selected = st.multiselect(
                    col,
                    options=options,
                    default=options,
                    key=f"{key_prefix}_filter_{col}",
                )
                filters[col] = set(selected) if selected else None

    active = sum(1 for v in filters.values() if v is not None)
    if active:
        st.caption(f"Filters active: {active} column(s). Export uses filtered data.")

    filtered = _apply_column_filters(df, {k: v for k, v in filters.items() if v is not None})

    # Sorting
    st.markdown("**Sort**")
    sc1, sc2 = st.columns(2)
    with sc1:
        sort_col = st.selectbox(
            "Sort by",
            options=df.columns.tolist(),
            key=f"{key_prefix}_sort_col",
        )
    with sc2:
        sort_asc = st.radio("Order", ["Ascending", "Descending"], horizontal=True, key=f"{key_prefix}_sort_order")
    ascending = sort_asc == "Ascending"
    filtered = _safe_sort(filtered, sort_col, ascending)

    st.download_button(
        f"Export {export_filename}",
        data=filtered.to_csv(index=False),
        file_name=export_filename,
        mime="text/csv",
        key=f"{key_prefix}_export",
    )

    st.dataframe(filtered, use_container_width=True, height=400)


def _show_tables_with_filters() -> None:
    """Display Daily and Weekly tables with per-column filters and export above each."""
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
