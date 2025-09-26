# 🏆 Swiss Weather Intelligence System - Judge Evaluation Guide
## LauzHack 2025 - Ready for Evaluation

---

## 🚀 **INSTANT SETUP - ONE COMMAND**

### **For Windows:**
```bash
# Double-click or run:
start_app.bat
```

### **For Linux/Mac:**
```bash
# Create and run setup script
chmod +x setup.sh
./setup.sh

# Or manual setup:
python -m venv .venv
source .venv/bin/activate
pip install streamlit pandas numpy matplotlib plotly seaborn requests scipy scikit-learn
streamlit run weather_app.py
```

**📍 Application URL:** http://localhost:8501


## 🛠️ **TROUBLESHOOTING**

### **If application doesn't start:**
```bash
# Check Python version (needs 3.8+)
python --version

# Manual environment setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run weather_app.py
```

### **If port 8501 is busy:**
```bash
streamlit run weather_app.py --server.port 8502
```

### **If dependencies fail:**
```bash
pip install --upgrade pip
pip install streamlit pandas numpy matplotlib plotly seaborn requests scipy scikit-learn
```
