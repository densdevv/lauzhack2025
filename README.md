## Also accessible publicly here: https://bias-and-variance.streamlit.app/

## Installation
*Important: Make sure to have Python installed before running!* 
- Simply run the start_app.bat file. The program will install all necessary libraries then start a localhost server.
- Manual (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run weather_app.py
```

- First installation takes around 4-5 minutes to download and install all necessary libraries.
- Once launched, please allow the app a few seconds to gather data and start predictions.

## Key Features
- Weather parameters (temperature, atmospheric pressure, precipitation, etc.) predictions based on previous data
- Personalized safety measures based on user background selection in control panel
- Extreme weather simulations in control panel
- AI-powered weather anomaly detection for extreme weather forecast with 7-days.

## Important Remarks
- The model trains itself on recent prediction accuracy. For better results, please give the app some time to catch up.
- The ML score is naturally lower for extreme weather scenarios, as they are rare and unpredictable. 
- For demonstration purposes, changes in weather conditions are exaggerated.
- Time is sped up by a factor of 300. 5 minutes pass every second, this is why the "current" time is actually the future.
- This project generates mock data as "actual data".
- You might have to reload app/site between extreme weather scenario switches to clear alert history.
