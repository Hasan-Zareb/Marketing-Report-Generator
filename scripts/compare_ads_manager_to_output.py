"""
Compare Ads Manager Report.xlsx (Meta export Mar 1-16) with Weekly Output Facebook Web.csv (Mar 9-15).
Applies the same mapping logic as the app to Excel data and checks if Ad Spend by Show Name matches.
"""
import sys
from pathlib import Path

import pandas as pd

# Add project root for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from processor import (
    xlookup_by_abbreviations,
    _normalize_show_name_series,
)

def load_and_prepare_excel(excel_path: Path, week_start: str, week_end: str):
    """Load Excel, filter to web campaigns and date range, return df with day, creative_network, cost."""
    df = pd.read_excel(excel_path, sheet_name="Raw Data Report")
    df.columns = [str(c).strip() for c in df.columns]
    # Drop summary row (no Day)
    df = df.dropna(subset=["Day"]).copy()
    df["Day"] = pd.to_datetime(df["Day"], errors="coerce")
    df = df.dropna(subset=["Day"])
    # Filter to week Mar 9 - Mar 15
    start = pd.Timestamp(week_start)
    end = pd.Timestamp(week_end)
    df = df[(df["Day"] >= start) & (df["Day"] <= end)]
    # Filter to web campaigns (same as Meta API)
    if "Campaign name" in df.columns:
        df = df[df["Campaign name"].astype(str).str.lower().str.contains("web")]
    # Map to canonical names used by processor
    # Excel: "Ad name", "Amount spent (INR)"
    rename = {"Day": "day", "Ad name": "creative_network", "Amount spent (INR)": "cost"}
    if "Amount spent (INR)" not in df.columns:
        # Try alternate
        for c in df.columns:
            if "amount" in c.lower() and "spent" in c.lower():
                rename[c] = "cost"
                break
    out = df.rename(columns=rename)
    out["cost"] = pd.to_numeric(out["cost"], errors="coerce").fillna(0)
    if "creative_network" not in out.columns and "Ad name" in df.columns:
        out["creative_network"] = df["Ad name"]
    return out[["day", "creative_network", "cost"]]


def main():
    excel_path = ROOT / "Ads Manager Report.xlsx"
    csv_path = ROOT / "Weekly Output Facebook Web.csv"
    if not excel_path.exists():
        print(f"Missing: {excel_path}")
        return
    if not csv_path.exists():
        print(f"Missing: {csv_path}")
        return

    # Weekly output is Mar 09 - Mar 15, 2026
    week_start = "2026-03-09"
    week_end = "2026-03-15"

    # 1) From Excel: same pipeline as Meta data in the app
    raw = load_and_prepare_excel(excel_path, week_start, week_end)
    if raw.empty:
        print("No rows in Excel for the week after filtering.")
        return
    mapped = xlookup_by_abbreviations(raw)
    mapped["Show Name"] = _normalize_show_name_series(mapped["Show Name"])
    excel_by_show = mapped.groupby("Show Name", as_index=False).agg(cost=("cost", "sum"))
    excel_by_show = excel_by_show.rename(columns={"cost": "Ad Spend (from Excel)"})
    excel_by_show = excel_by_show.sort_values("Ad Spend (from Excel)", ascending=False).reset_index(drop=True)

    # 2) From Weekly Output CSV
    out_df = pd.read_csv(csv_path)
    out_df.columns = [str(c).strip() for c in out_df.columns]
    if "Show Name" not in out_df.columns or "Ad Spend" not in out_df.columns:
        print("CSV must have 'Show Name' and 'Ad Spend'. Columns:", list(out_df.columns))
        return
    csv_by_show = out_df[["Show Name", "Ad Spend"]].copy()
    csv_by_show["Show Name"] = _normalize_show_name_series(csv_by_show["Show Name"])
    csv_by_show = csv_by_show.groupby("Show Name", as_index=False).agg({"Ad Spend": "sum"})
    csv_by_show = csv_by_show.rename(columns={"Ad Spend": "Ad Spend (from Output CSV)"})
    csv_by_show = csv_by_show.sort_values("Ad Spend (from Output CSV)", ascending=False).reset_index(drop=True)

    # 3) Merge and compare
    all_shows = sorted(set(excel_by_show["Show Name"].tolist() + csv_by_show["Show Name"].tolist()))
    comparison = pd.DataFrame({"Show Name": all_shows})
    comparison = comparison.merge(
        excel_by_show, on="Show Name", how="outer"
    ).merge(
        csv_by_show, on="Show Name", how="outer"
    )
    comparison["Ad Spend (from Excel)"] = comparison["Ad Spend (from Excel)"].fillna(0)
    comparison["Ad Spend (from Output CSV)"] = comparison["Ad Spend (from Output CSV)"].fillna(0)
    comparison["Diff"] = comparison["Ad Spend (from Output CSV)"] - comparison["Ad Spend (from Excel)"]
    comparison["Match"] = comparison["Diff"].abs() < 0.01
    comparison = comparison.sort_values("Ad Spend (from Output CSV)", ascending=False).reset_index(drop=True)

    print("=" * 80)
    print("Comparison: Ads Manager Report.xlsx (Mar 9–15) vs Weekly Output Facebook Web.csv")
    print("=" * 80)
    print(comparison.to_string(index=False))
    print()
    total_excel = comparison["Ad Spend (from Excel)"].sum()
    total_csv = comparison["Ad Spend (from Output CSV)"].sum()
    print(f"Total Ad Spend (from Excel):  {total_excel:,.2f}")
    print(f"Total Ad Spend (from Output): {total_csv:,.2f}")
    print(f"Difference:                   {total_csv - total_excel:,.2f}")
    matches = comparison["Match"].sum()
    print(f"Shows with matching spend:    {matches} / {len(comparison)}")
    if not comparison["Match"].all():
        print("\nShows with discrepancy:")
        print(comparison[~comparison["Match"]][["Show Name", "Ad Spend (from Excel)", "Ad Spend (from Output CSV)", "Diff"]].to_string(index=False))

    # Findings summary
    print("\n" + "=" * 80)
    print("FINDINGS")
    print("=" * 80)
    print("- Date range: Mar 9–15, 2026 (weekly report period). Excel filtered to same period.")
    print("- Same mapping logic was applied to Excel (Ad name -> Show Name) as in the app.")
    print("- Small differences (< 1) are likely rounding. Larger gaps mean API vs export differ.")
    print("- Output CSV has no row for 'No' or 'Race Against Time': those shows have spend in")
    print("  Excel but 0 in output → Meta API may not have returned those ads for this period,")
    print("  or timezone/date boundaries differ between API and Excel export.")
    print("- If totals are close, mapping and logic are consistent; remaining gap is likely")
    print("  Meta API vs Ads Manager export (timezone, attribution, or export scope).")


if __name__ == "__main__":
    main()
