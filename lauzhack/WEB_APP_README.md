# 🇨🇭 Swiss Weather Intelligence System
## LauzHack 2025 - Judge Evaluation Ready

### � Advanced Weather Intelligence with ML-Enhanced Predictions

A cutting-edge web application delivering personalized weather intelligence for Switzerland, featuring machine learning-based predictions, emergency scenario simulation, and professional-grade risk assessment.

---

## 🚀 **QUICK START FOR JUDGES**

### **Easiest Method - One-Click Launch:**
```bash
# Windows: Double-click this file
start_app.bat

# Or run in terminal:
./start_app.bat
```

The application will:
- ✅ Auto-create Python virtual environment
- ✅ Install all dependencies automatically  
- ✅ Launch web interface at http://localhost:8501
- ✅ Open in your default browser

### **Manual Setup (if needed):**
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies (optimized order for faster startup)
pip install streamlit pandas numpy plotly requests
pip install scipy scikit-learn matplotlib seaborn

# Or install all at once using requirements.txt
pip install -r requirements.txt

# Run application
streamlit run weather_app.py
```