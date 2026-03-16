"""Streamlit UI: upload input CSV or fetch from Adjust API, generate Daily/Weekly reports, dashboard."""

import io
import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from adjust_client import fetch_adjust_report
from meta_ads_client import fetch_all_accounts as fetch_meta_ads, filter_by_date as meta_filter_by_date
from processor import (
    CANONICAL_COLUMNS,
    NEW_TO_CANONICAL,
    export_csv,
    get_daily_by_show_all,
    process,
    process_facebook_web,
    validate_columns,
)

# Plotly dark theme and color palette (purple, green, orange accents)
PLOTLY_TEMPLATE = "plotly_dark"
CHART_COLORS = ["#9D4EDD", "#2D6A4F", "#E85D04", "#7209B7", "#06D6A0"]


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Marketing Report Generator", layout="wide")
    _init_session_state()
    _inject_dashboard_css()

    # Header: title, date range, Reset
    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown("## ◆ Marketing Report Generator")
        report_range = st.session_state.report_date_range
        if report_range:
            start, end = report_range
            st.caption(f"**Report period:** {start.strftime('%b %d, %Y')} – {end.strftime('%b %d, %Y')}")
    with h2:
        if st.button("Reset", type="secondary"):
            for key in [
                "daily_csv", "weekly_csv", "daily_df", "weekly_df",
                "daily_by_show_df", "report_date_range", "last_file_id",
                "adjust_fetched_df", "meta_fetched_df",
                "daily_web_df", "weekly_web_df", "daily_web_csv", "weekly_web_csv",
                "fb_web_status",
            ]:
                if key in st.session_state:
                    st.session_state[key] = None
            st.rerun()

    # Tabs: Report Generator first (default), then Dashboard
    tab_report_gen, tab_dashboard = st.tabs(["Report Generator", "Dashboard"])
    with tab_report_gen:
        _render_report_generator()
    with tab_dashboard:
        _render_dashboard()


def _inject_dashboard_css() -> None:
    """Inject CSS for rounded KPI cards and dashboard styling."""
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            background-color: rgba(30, 33, 40, 0.6);
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(157, 78, 221, 0.2);
        }
        [data-testid="stMetric"] label {
            color: #B4A5D9 !important;
            font-size: 0.75rem !important;
            text-transform: uppercase;
        }
        .by-show-section { margin: 1rem 0; }
        .by-show-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 1rem;
            margin-top: 0.75rem;
        }
        .by-show-card {
            background: rgba(30, 33, 40, 0.8);
            border: 1px solid rgba(157, 78, 221, 0.25);
            border-radius: 0.5rem;
            padding: 1rem;
            transition: border-color 0.15s ease;
        }
        .by-show-card:hover { border-color: rgba(157, 78, 221, 0.5); }
        .by-show-name {
            font-weight: 600;
            font-size: 0.95rem;
            color: #FAFAFA;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .by-show-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .by-show-funnel-wrap {
            margin: 0.5rem 0;
            font-size: 0.7rem;
            color: #9CA3AF;
        }
        .by-show-funnel-bar {
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            display: flex;
            margin-top: 0.25rem;
            background: rgba(255,255,255,0.06);
        }
        .by-show-funnel-seg { height: 100%; min-width: 2px; }
        .by-show-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem 1rem;
            margin-top: 0.6rem;
            font-size: 0.8rem;
            color: #D1D5DB;
        }
        .by-show-metrics span { white-space: nowrap; }
        .by-show-metrics .label { color: #9CA3AF; }
        .by-show-metrics .val { font-weight: 600; color: #E5E7EB; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    if "daily_csv" not in st.session_state:
        st.session_state.daily_csv = None
    if "weekly_csv" not in st.session_state:
        st.session_state.weekly_csv = None
    if "daily_df" not in st.session_state:
        st.session_state.daily_df = None
    if "weekly_df" not in st.session_state:
        st.session_state.weekly_df = None
    if "daily_by_show_df" not in st.session_state:
        st.session_state.daily_by_show_df = None
    if "report_date_range" not in st.session_state:
        st.session_state.report_date_range = None
    if "last_file_id" not in st.session_state:
        st.session_state.last_file_id = None
    if "dashboard_filter" not in st.session_state:
        st.session_state.dashboard_filter = "This Week"
    if "adjust_fetched_df" not in st.session_state:
        st.session_state.adjust_fetched_df = None
    if "meta_fetched_df" not in st.session_state:
        st.session_state.meta_fetched_df = None
    if "daily_web_df" not in st.session_state:
        st.session_state.daily_web_df = None
    if "weekly_web_df" not in st.session_state:
        st.session_state.weekly_web_df = None
    if "daily_web_csv" not in st.session_state:
        st.session_state.daily_web_csv = None
    if "weekly_web_csv" not in st.session_state:
        st.session_state.weekly_web_csv = None
    if "fb_web_status" not in st.session_state:
        st.session_state.fb_web_status = None


def _render_dashboard() -> None:
    """Dashboard tab: KPIs, By Show funnel, 2×2 charts (day-by-day)."""
    df = st.session_state.daily_by_show_df
    if df is None:
        st.info("Generate reports in the **Report Generator** tab (upload CSV and click **Generate reports**) to see the dashboard.")
        return
    if df.empty:
        st.warning("No data for the selected report period. Try a different date range in the **Report Generator** tab.")
        return

    report_range = st.session_state.report_date_range
    if not report_range:
        st.warning("No report date range set. Generate reports in the Report Generator tab first.")
        return
    start, end = report_range

    # Quick filter: which dates to show in dashboard
    filter_choice = st.radio(
        "Date range",
        ["This Week", "Yesterday", "Today", "All"],
        horizontal=True,
        key="dashboard_filter_radio",
    )
    st.session_state.dashboard_filter = filter_choice

    # Filter daily_by_show_df by selected range
    df = df.copy()
    df["day"] = pd.to_datetime(df["day"]).dt.date
    if filter_choice == "This Week":
        # Last 7 days of the data range
        week_end = end
        week_start = end - timedelta(days=6)
        df = df[(df["day"] >= week_start) & (df["day"] <= week_end)]
    elif filter_choice == "Yesterday":
        yesterday = date.today() - timedelta(days=1)
        df = df[df["day"] == yesterday]
    elif filter_choice == "Today":
        df = df[df["day"] == date.today()]
    # "All" = no filter (full dataset)

    if df.empty:
        st.warning("No data for the selected range.")
        return

    _render_kpi_row(df)
    st.divider()
    _render_by_show_funnel(df)
    st.divider()
    _render_chart_grid(df)


def _render_kpi_row(df: pd.DataFrame) -> None:
    """Five KPI cards: Installs, Free Trials, Paid Subs, Ad Spend, Avg CAC."""
    total_installs = df["#Installs"].sum() if "#Installs" in df.columns else 0
    total_trials = df["#Free Trials"].sum()
    total_subs = df["#Subscriptions"].sum()
    total_spend = df["Ad Spend"].sum()
    cac_vals = pd.to_numeric(df["CAC"], errors="coerce").dropna()
    avg_cac = cac_vals.mean() if len(cac_vals) else None

    def _fmt_num(x):
        if x >= 1_000_000:
            return f"{x / 1_000_000:.1f}M"
        if x >= 1_000:
            return f"{x / 1_000:.1f}K"
        return f"{x:.0f}"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("INSTALLS", _fmt_num(total_installs))
    with col2:
        st.metric("FREE TRIALS", _fmt_num(total_trials))
    with col3:
        st.metric("PAID SUBS", _fmt_num(total_subs))
    with col4:
        st.metric("AD SPEND", _fmt_num(total_spend))
    with col5:
        st.metric("AVG CAC", f"{avg_cac:.1f}" if avg_cac is not None and not pd.isna(avg_cac) else "–")


def _fmt_short(value: float) -> str:
    """Format number for display (e.g. 1100682 -> 11L, 8454 -> 8.5K)."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}L"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def _render_by_show_funnel(df: pd.DataFrame) -> None:
    """By Show section: card grid with mini funnel bars and clear metrics."""
    st.subheader("By Show")
    st.caption("Funnel: Installs → Free Trials → Paid. Hover cards for focus.")
    agg_dict = {"#Free Trials": "sum", "#Subscriptions": "sum", "Ad Spend": "sum"}
    if "#Installs" in df.columns:
        agg_dict["#Installs"] = "sum"
    agg = df.groupby("Show Name", as_index=False).agg(agg_dict)
    if "#Installs" not in agg.columns:
        agg["#Installs"] = 0
    # CAC and I→Paid %
    agg["CAC"] = (agg["Ad Spend"] / agg["#Subscriptions"]).where(agg["#Subscriptions"] > 0, float("nan"))
    agg["I_to_Paid_pct"] = (agg["#Subscriptions"] / agg["#Installs"] * 100).where(agg["#Installs"] > 0, float("nan"))
    sort_col = "#Installs" if "#Installs" in agg.columns else "#Free Trials"
    agg = agg.sort_values(sort_col, ascending=False).head(16)

    # Distinct colors per card (cycle through palette)
    dot_colors = ["#9D4EDD", "#2D6A4F", "#E85D04", "#06D6A0", "#7209B7", "#F72585", "#4CC9F0", "#4361EE"]

    cards_html = []
    for i, row in agg.iterrows():
        show_name = str(row["Show Name"])
        show_name_esc = show_name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        inst, trials, paid = row["#Installs"], row["#Free Trials"], row["#Subscriptions"]
        spend = row["Ad Spend"]
        total = inst + trials + paid
        if total > 0:
            w_inst = (inst / total) * 100
            w_trials = (trials / total) * 100
            w_paid = (paid / total) * 100
        else:
            w_inst = w_trials = w_paid = 33.33
        cac_val = row["CAC"]
        i2p = row["I_to_Paid_pct"]
        cac_str = f"{cac_val:,.0f}" if pd.notna(cac_val) and cac_val == cac_val else "–"
        i2p_str = f"{i2p:.1f}%" if pd.notna(i2p) and i2p == i2p else "–"
        dot = dot_colors[i % len(dot_colors)]
        card = (
            f'<div class="by-show-card">'
            f'<div class="by-show-name"><span class="by-show-dot" style="background:{dot}"></span>{show_name_esc}</div>'
            f'<div class="by-show-funnel-wrap">'
            f'<span>Installs &#183; Trials &#183; Paid</span>'
            f'<div class="by-show-funnel-bar">'
            f'<span class="by-show-funnel-seg" style="width:{w_inst}%;background:#9D4EDD"></span>'
            f'<span class="by-show-funnel-seg" style="width:{w_trials}%;background:#2D6A4F"></span>'
            f'<span class="by-show-funnel-seg" style="width:{w_paid}%;background:#E85D04"></span>'
            f'</div></div>'
            f'<div class="by-show-metrics">'
            f'<span><span class="label">Installs</span> <span class="val">{_fmt_short(inst)}</span></span>'
            f'<span><span class="label">Trials</span> <span class="val">{_fmt_short(trials)}</span></span>'
            f'<span><span class="label">Paid</span> <span class="val">{_fmt_short(paid)}</span></span>'
            f'<span><span class="label">Spend</span> <span class="val">{_fmt_short(spend)}</span></span>'
            f'<span><span class="label">I&#8594;Paid</span> <span class="val">{i2p_str}</span></span>'
            f'<span><span class="label">CAC</span> <span class="val">{cac_str}</span></span>'
            f'</div></div>'
        )
        cards_html.append(card)

    # Render cards in a grid: each card in its own st.markdown() to avoid Streamlit
    # truncating or failing to parse one large HTML blob
    n_cols = 4
    for row_start in range(0, len(cards_html), n_cols):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            idx = row_start + j
            if idx < len(cards_html):
                with cols[j]:
                    st.markdown(cards_html[idx], unsafe_allow_html=True)


def _render_chart_grid(df: pd.DataFrame) -> None:
    """2×2 grid: Installs by Show, Conversion rates, CAC by show, Daily volume / Ad Spend."""
    df = df.copy()
    df["day"] = pd.to_datetime(df["day"])

    # Show selector for Installs by Show and CAC by Show (default: All)
    all_shows = sorted(df["Show Name"].unique().tolist())
    show_options = ["All"] + all_shows
    selected_shows = st.multiselect(
        "Shows to display in **Installs by Show** and **CAC by Show** (default: All). Pick one or more, or All.",
        options=show_options,
        default=["All"],
        key="dashboard_shows_multiselect",
    )
    if "All" in selected_shows or not selected_shows:
        df_installs_cac = df
    else:
        df_installs_cac = df[df["Show Name"].isin(selected_shows)].copy()

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        if "#Installs" in df.columns:
            st.plotly_chart(_chart_installs_by_show(df_installs_cac), use_container_width=True)
        else:
            st.plotly_chart(_chart_daily_volume(df), use_container_width=True)
    with row1_col2:
        st.plotly_chart(_chart_conversion_rates(df), use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.plotly_chart(_chart_cac_by_show(df_installs_cac), use_container_width=True)
    with row2_col2:
        st.plotly_chart(_chart_daily_spend(df), use_container_width=True)


def _chart_installs_by_show(df: pd.DataFrame):
    if "#Installs" not in df.columns or df.empty:
        return go.Figure().update_layout(template=PLOTLY_TEMPLATE, title="Installs by Show")
    wide = df.pivot_table(index="day", columns="Show Name", values="#Installs", aggfunc="sum", fill_value=0)
    fig = go.Figure()
    for i, col in enumerate(wide.columns):
        fig.add_trace(
            go.Scatter(
                x=wide.index,
                y=wide[col],
                name=col,
                mode="lines",
                line=dict(width=1.5),
            )
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="INSTALLS BY SHOW",
        xaxis_title="Date",
        yaxis_title="Installs",
        colorway=CHART_COLORS,
        legend=dict(traceorder="normal"),
    )
    return fig


def _chart_conversion_rates(df: pd.DataFrame):
    if df.empty:
        return go.Figure().update_layout(template=PLOTLY_TEMPLATE, title="Conversion Rates")
    agg_cols = {"#Free Trials": "sum", "#Subscriptions": "sum"}
    if "#Installs" in df.columns:
        agg_cols["#Installs"] = "sum"
    daily = df.groupby("day", as_index=False).agg(agg_cols)
    daily["Conv %"] = (daily["#Subscriptions"] / daily["#Free Trials"] * 100).where(daily["#Free Trials"] > 0, float("nan"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["Conv %"], name="Trial → Sub %", mode="lines+markers"))
    if "#Installs" in daily.columns and daily["#Installs"].sum() > 0:
        daily["Conv Install %"] = (daily["#Free Trials"] / daily["#Installs"] * 100).where(daily["#Installs"] > 0, float("nan"))
        fig.add_trace(go.Scatter(x=daily["day"], y=daily["Conv Install %"], name="Install → Trial %", mode="lines+markers"))
    fig.update_layout(template=PLOTLY_TEMPLATE, title="CONVERSION RATES", xaxis_title="Date", yaxis_title="%", colorway=CHART_COLORS)
    return fig


def _chart_cac_by_show(df: pd.DataFrame):
    if df.empty or "CAC" not in df.columns:
        return go.Figure().update_layout(template=PLOTLY_TEMPLATE, title="CAC by Show")
    df = df.copy()
    df["CAC_num"] = pd.to_numeric(df["CAC"], errors="coerce")
    df = df.dropna(subset=["CAC_num"])
    if df.empty:
        return go.Figure().update_layout(template=PLOTLY_TEMPLATE, title="CAC by Show")
    fig = px.line(df, x="day", y="CAC_num", color="Show Name", title="CAC BY SHOW (day by day)")
    fig.update_layout(template=PLOTLY_TEMPLATE, colorway=CHART_COLORS, legend=dict(traceorder="normal"))
    fig.update_traces(mode="lines", line=dict(width=1.5))
    return fig


def _chart_daily_volume(df: pd.DataFrame):
    if df.empty:
        return go.Figure().update_layout(template=PLOTLY_TEMPLATE, title="Daily Volume")
    daily = df.groupby("day", as_index=False).agg({"#Free Trials": "sum", "#Subscriptions": "sum"})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["#Free Trials"], name="Free Trials", fill="tozeroy"))
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["#Subscriptions"], name="Subscriptions", fill="tozeroy"))
    fig.update_layout(template=PLOTLY_TEMPLATE, title="DAILY VOLUME", xaxis_title="Date", yaxis_title="Count", colorway=CHART_COLORS)
    return fig


def _chart_daily_spend(df: pd.DataFrame):
    if df.empty:
        return go.Figure().update_layout(template=PLOTLY_TEMPLATE, title="Daily Ad Spend")
    daily = df.groupby("day", as_index=False).agg({"Ad Spend": "sum"})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["Ad Spend"], name="Ad Spend", fill="tozeroy", line=dict(color="#E85D04")))
    fig.update_layout(template=PLOTLY_TEMPLATE, title="DAILY AD SPEND", xaxis_title="Date", yaxis_title="Ad Spend", colorway=CHART_COLORS)
    return fig


def _render_report_generator() -> None:
    """Report Generator tab: upload CSV or fetch from Adjust API, select dates, generate reports, export."""
    st.subheader("Report Generator")

    data_source = st.radio(
        "Data source",
        ["Fetch from Adjust & Meta API", "Upload CSV"],
        horizontal=True,
        key="report_data_source",
    )

    df = None

    if data_source == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload input CSV (e.g. Marketing Performance Dec 2025 to Jan 2026.csv)",
            type=["csv"],
            help="Old format: creative_network, day, free_trial_be_events, revenue_3ea7d4b1_events, cost. "
            "New format: Ad name, Day (date), FREE_TRIAL_BE, REVENUE, Ad spend.",
            key="deep_dive_upload",
        )
        if uploaded is not None:
            current_file_id = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.last_file_id != current_file_id:
                st.session_state.daily_csv = None
                st.session_state.weekly_csv = None
                st.session_state.daily_df = None
                st.session_state.weekly_df = None
                st.session_state.daily_by_show_df = None
                st.session_state.report_date_range = None
                st.session_state.last_file_id = current_file_id
            try:
                raw = uploaded.read()
                df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
            except Exception:
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
                    except Exception as e2:
                        st.error(f"Could not read CSV with latin-1: {e2}")
                        return
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
                    return
            df.columns = [c.lstrip("\ufeff") if isinstance(c, str) else c for c in df.columns]
        else:
            st.info("Upload an input CSV to continue.")
            if st.session_state.daily_df is not None and st.session_state.weekly_df is not None:
                st.info("Previous reports are available below. Upload a new file to generate new reports.")
                _show_tables_with_filters()
            return

    else:
        st.caption("Fetch Marketing Performance data from Adjust (partners 34 & 254) and Facebook Web spend from Meta Ads Manager in one go.")
        load_dotenv()
        if st.button("Fetch data from Adjust & Meta"):
            api_token = os.getenv("ADJUST_API_TOKEN")
            if not api_token or not api_token.strip():
                st.error("Set ADJUST_API_TOKEN in your .env file to use the Adjust API.")
                return

            fetch_end = date.today() - timedelta(days=1)
            fetch_start = fetch_end - timedelta(days=365 * 2)

            with st.spinner("Fetching data from Adjust API…"):
                try:
                    fetched = fetch_adjust_report(api_token, fetch_start, fetch_end)
                    st.session_state.adjust_fetched_df = fetched
                except Exception as e:
                    st.error(f"Adjust API error: {e}")
                    return

            meta_token = os.getenv("META_ACCESS_TOKEN", "").strip()
            if meta_token and meta_token not in ("PASTE_YOUR_TOKEN_HERE", "your_meta_token_here"):
                with st.spinner("Fetching Facebook Web spend from Meta Ads API…"):
                    try:
                        meta_full = fetch_meta_ads(access_token=meta_token)
                        st.session_state.meta_fetched_df = meta_full
                        st.session_state.fb_web_status = f"Meta: {len(meta_full)} rows fetched."
                    except Exception as e:
                        st.session_state.meta_fetched_df = None
                        st.session_state.fb_web_status = f"Meta Ads API error: {e}"
                        st.warning(f"Meta Ads fetch failed: {e}. Adjust data is still available.")
            else:
                st.session_state.meta_fetched_df = None
                st.session_state.fb_web_status = "META_ACCESS_TOKEN not set. Facebook Web reports will be skipped."

            adjust_count = len(st.session_state.adjust_fetched_df)
            meta_count = len(st.session_state.meta_fetched_df) if st.session_state.meta_fetched_df is not None else 0
            st.success(f"Fetched {adjust_count:,} rows from Adjust, {meta_count:,} rows from Meta. Choose dates below and click Generate reports.")

        if st.session_state.adjust_fetched_df is not None:
            df = st.session_state.adjust_fetched_df
        else:
            st.info("Click **Fetch data from Adjust & Meta** to load data, then choose report dates and generate reports below.")
            if st.session_state.daily_df is not None and st.session_state.weekly_df is not None:
                st.info("Previous reports are available below.")
                _show_tables_with_filters()
            return

    missing = validate_columns(df)
    if missing:
        st.error(
            f"Missing required columns: {', '.join(missing)}. "
            f"Need either old format ({', '.join(CANONICAL_COLUMNS)}) or "
            f"new format ({', '.join(NEW_TO_CANONICAL)})."
        )
        return

    if df.empty:
        st.warning("No data to process (empty file or no rows from Adjust).")
        return

    st.subheader("Select Report Dates")
    col1, col2 = st.columns(2)
    with col1:
        daily_date = st.date_input(
            "Daily Report Date",
            value=(pd.Timestamp.now() - pd.Timedelta(days=1)).date(),
            help="Select the date for the daily report",
            key="daily_date",
        )
    with col2:
        weekly_start_date = st.date_input(
            "Weekly Report Start Date",
            value=(pd.Timestamp.now() - pd.Timedelta(days=7)).date(),
            help="Select the start date for the weekly report (7 days will be included from this date)",
            key="weekly_start_date",
        )
    weekly_end_date = weekly_start_date + timedelta(days=6)
    st.caption(f"Weekly report will include dates: {weekly_start_date.strftime('%b %d, %Y')} - {weekly_end_date.strftime('%b %d, %Y')}")

    if st.button("Generate reports"):
        with st.spinner("Generating reports and dashboard data…"):
            try:
                daily, weekly, _ = process(
                    input_df=df,
                    daily_date=pd.Timestamp(daily_date),
                    weekly_start_date=pd.Timestamp(weekly_start_date),
                )
                daily_by_show_full = get_daily_by_show_all(df)
            except Exception as e:
                st.error(f"Processing failed: {e}")
                raise

        st.session_state.daily_df = daily
        st.session_state.weekly_df = weekly
        st.session_state.daily_by_show_df = daily_by_show_full
        if not daily_by_show_full.empty:
            days = pd.to_datetime(daily_by_show_full["day"])
            st.session_state.report_date_range = (days.min().date(), days.max().date())
        else:
            st.session_state.report_date_range = (weekly_start_date, weekly_start_date + timedelta(days=6))
        st.session_state.daily_csv = export_csv(daily)
        st.session_state.weekly_csv = export_csv(weekly)

        # --- Facebook Web report: Adjust (trials/subs) + Meta (spend) ---
        meta_full = st.session_state.get("meta_fetched_df")
        if meta_full is not None and not meta_full.empty:
            date_start = min(daily_date, weekly_start_date)
            date_end = max(daily_date, weekly_end_date)
            meta_filtered = meta_filter_by_date(meta_full, str(date_start), str(date_end))
            st.session_state.fb_web_status = f"Meta: {len(meta_full)} total rows, {len(meta_filtered)} in selected date range."

            if not meta_filtered.empty:
                try:
                    daily_web, weekly_web = process_facebook_web(
                        adjust_df=df,
                        meta_df=meta_filtered,
                        daily_date=pd.Timestamp(daily_date),
                        weekly_start_date=pd.Timestamp(weekly_start_date),
                    )
                    st.session_state.daily_web_df = daily_web
                    st.session_state.weekly_web_df = weekly_web
                    st.session_state.daily_web_csv = export_csv(daily_web)
                    st.session_state.weekly_web_csv = export_csv(weekly_web)
                except Exception as e:
                    st.session_state.fb_web_status = f"Facebook Web report generation failed: {e}"
                    st.session_state.daily_web_df = None
                    st.session_state.weekly_web_df = None
            else:
                st.session_state.fb_web_status = "No Meta Ads data for the selected date range."
                st.session_state.daily_web_df = None
                st.session_state.weekly_web_df = None
        else:
            if meta_full is None:
                st.session_state.fb_web_status = st.session_state.get("fb_web_status") or "Meta data not fetched. Click 'Fetch data from Adjust & Meta' first."
            else:
                st.session_state.fb_web_status = "Meta API returned no data."
            st.session_state.daily_web_df = None
            st.session_state.weekly_web_df = None

        st.success("Reports ready. View and export below.")
        st.rerun()

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
    """Display Daily and Weekly tables, plus Facebook Web tables if available."""
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

    # Facebook Web ads section (always show header; show tables or status message)
    st.divider()
    st.markdown("### Facebook Web Ads")
    st.caption("Trials & subscriptions from Adjust (web campaigns) | Ad spend from Meta Ads Manager")

    daily_web = st.session_state.get("daily_web_df")
    weekly_web = st.session_state.get("weekly_web_df")
    meta_token = os.getenv("META_ACCESS_TOKEN", "").strip()

    if daily_web is not None and weekly_web is not None and not daily_web.empty and not weekly_web.empty:
        _table_with_filters(
            daily_web,
            "Daily Output (Facebook Web)",
            "Daily Output (Facebook Web).csv",
            "daily_web",
        )
        st.divider()
        _table_with_filters(
            weekly_web,
            "Weekly Output (Facebook Web)",
            "Weekly Output (Facebook Web).csv",
            "weekly_web",
        )
    elif daily_web is not None and weekly_web is not None:
        st.info("Facebook Web reports were generated but contain no data for the selected dates.")
    else:
        fb_status = st.session_state.get("fb_web_status")
        if fb_status:
            st.info(fb_status)
        elif not meta_token or meta_token in ("PASTE_YOUR_TOKEN_HERE", "your_meta_token_here"):
            st.info("Set META_ACCESS_TOKEN in .env (or Streamlit Secrets) to enable Facebook Web reports.")
        else:
            st.info("Click **Generate reports** to produce Facebook Web reports alongside the main reports.")


if __name__ == "__main__":
    main()
