@echo off
echo ================================================================
echo 🇨🇭 SWISS WEATHER INTELLIGENCE SYSTEM - LauzHack 2025
echo ================================================================
echo.
echo 🚀 Starting Advanced Weather Intelligence Web Application...
echo.
echo 🎯 Key Features:
echo    • ML-Enhanced Weather Prediction System
echo    • Personalized Risk Assessment (12 Professional Backgrounds)
echo    • Emergency Scenario Simulation (Heat Wave, Storm, Flood)
echo    • Real-time Swiss Weather Monitoring
echo    • Extended 7-Day Extreme Weather Forecasting
echo.
echo 📋 Judge Evaluation Ready:
echo    • Pure ML-based predictions (no hard-coded limits)
echo    • Data-driven extreme weather forecasting
echo    • Professional-grade user experience
echo.
echo.
echo 🔧 Checking Python virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo ❌ Virtual environment not found! Creating new environment...
    python -m venv .venv
    echo ✅ Virtual environment created
)

echo.
echo 📦 Installing required dependencies...
.venv\Scripts\pip.exe install streamlit pandas numpy matplotlib plotly seaborn requests scipy scikit-learn

echo.
echo 🌐 Starting Streamlit Web Application...
echo 📍 URL: http://localhost:8501
echo.
echo 🛑 Press Ctrl+C to stop the application
echo ================================================================
echo.

.venv\Scripts\streamlit.exe run weather_app.py --server.headless false --server.port 8501 --server.address localhost

echo.
echo 🔄 Application stopped. Press any key to exit...
pause