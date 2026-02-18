# Marketing Report Generator

Creates **Daily** and **Weekly** marketing reports for Alright TV shows. Fetches data from the Adjust API (Marketing Performance report) or accepts an uploaded CSV, maps ad creatives to show names via hardcoded abbreviations, then outputs pivoted reports with #Free Trials, #Subscriptions, Ad Spend, Cost of Free Trial, and CAC.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Adjust API token:

```
ADJUST_API_TOKEN=your_token_here
```

## Run

```bash
streamlit run app.py
```

(or `python3 -m streamlit run app.py` if `streamlit` is not on your PATH)

Choose dates, click **Refresh – Fetch from Adjust & Generate Reports**, then download **Daily Output.csv** and **Weekly Output.csv**. You can also upload a CSV instead of using the API.

## About

Creates Daily and Weekly marketing reports of the Alright TV shows.
