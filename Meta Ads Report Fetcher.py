"""
Standalone runner for Meta Ads data.
Core fetch logic lives in meta_ads_client.py (shared with the Streamlit app).

Usage:
  1. Create a .env file with META_ACCESS_TOKEN=your_token_here
  2. pip install requests pandas python-dotenv
  3. python "Meta Ads Report Fetcher.py"
"""

from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from meta_ads_client import fetch_all_accounts, filter_by_date


if __name__ == "__main__":
    print("=" * 60)
    print("META ADS REPORT FETCHER")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Active filter: Campaign name contains 'web'")
    print("=" * 60)

    df = fetch_all_accounts(use_async=False)

    if df.empty:
        print("No data to save.")
    else:
        output_path = "meta_ads_report.csv"
        df.to_csv(output_path, index=False)
        print(f"\nFull report saved to: {output_path}")
        print(f"   Total rows:        {len(df)}")
        print(f"   Date range:        {df['Day'].min().date()} -> {df['Day'].max().date()}")
        print(f"   Total spend:       {df['Amount Spent'].sum():,.2f}")
        print(f"   Total impressions: {df['Impressions'].sum():,}")
        if "Link Clicks" in df.columns:
            print(f"   Total link clicks: {df['Link Clicks'].sum():,}")

        print("\n--- Example: Filter to March 2026 ---")
        filtered = filter_by_date(df, "2026-03-01", "2026-03-16")
        if not filtered.empty:
            print(filtered[["Day", "Campaign Name", "Ad Set Name", "Impressions", "Amount Spent"]].head(10).to_string())
        else:
            print("No data in that range.")
