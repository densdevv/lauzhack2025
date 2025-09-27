## Swiss Weather Intelligence — How to run

Quick options (Windows):

1) One-click (recommended)
- Double-click `start_app.bat` in the `lauzhack` folder
- It will create a virtual env, install deps, and open http://localhost:8501

2) Manual (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run weather_app.py
```

Notes
- If the browser doesn’t open automatically, visit http://localhost:8501
- To stop the app, press Ctrl+C in the terminal