# ⚽ WC 2026 Predictor

A Poisson + Elo + Dixon-Coles football prediction app for FIFA World Cup 2026.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud (free)
1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set "Main file path" to `app.py`
5. Click Deploy — live in ~2 minutes

## Files needed
- `app.py` — the app
- `group_match.csv` — group stage fixtures
- `knokout_match.csv` — knockout fixtures
- `requirements.txt`
