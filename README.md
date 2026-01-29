# Marketing Report Generator

Generates **Daily Output** and **Weekly Output** CSVs from an input CSV (e.g. `Input File 1.csv`). Uses `Show Name Reference.csv` to map `creative_network` codes to show names, then pivots by show and computes #Free Trials, #Subscriptions, Ad Spend, Cost of Free Trial, and CAC.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 -m streamlit run app.py
```

If `streamlit` is not on your PATH (e.g. you see "command not found"), use `python3 -m streamlit` as above.

Upload an input CSV, click **Generate reports**, then download **Daily Output.csv** and **Weekly Output.csv**.
