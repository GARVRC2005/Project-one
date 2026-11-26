@echo off
REM One-click setup + launch on Windows
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
if not exist data\newsimpact.db python -m newsimpact.cli build
streamlit run dashboard\app.py
