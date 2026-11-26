#!/usr/bin/env bash
set -e
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
[ -f data/newsimpact.db ] || python -m newsimpact.cli build
streamlit run dashboard/app.py
