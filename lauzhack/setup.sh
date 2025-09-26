#!/bin/bash
# Swiss Weather Intelligence System - Setup Script
# LauzHack 2025 - Judge Evaluation Ready

echo "================================================================"
echo "🇨🇭 SWISS WEATHER INTELLIGENCE SYSTEM - LauzHack 2025"
echo "================================================================"
echo ""
echo "🚀 Starting Advanced Weather Intelligence Web Application..."
echo ""
echo "🎯 Key Features:"
echo "   • ML-Enhanced Weather Prediction System"
echo "   • Personalized Risk Assessment (12 Professional Backgrounds)"
echo "   • Emergency Scenario Simulation (Heat Wave, Storm, Flood)"
echo "   • Real-time Swiss Weather Monitoring"
echo "   • Extended 7-Day Extreme Weather Forecasting"
echo ""
echo "📋 Judge Evaluation Ready:"
echo "   • Pure ML-based predictions (no hard-coded limits)"
echo "   • Data-driven extreme weather forecasting"
echo "   • Professional-grade user experience"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "🔧 Setting up Python virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate

echo ""
echo "📦 Installing required dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "🌐 Starting Streamlit Web Application..."
echo "📍 URL: http://localhost:8501"
echo ""
echo "🛑 Press Ctrl+C to stop the application"
echo "================================================================"
echo ""

# Start the application
streamlit run weather_app.py --server.headless false --server.port 8501 --server.address localhost