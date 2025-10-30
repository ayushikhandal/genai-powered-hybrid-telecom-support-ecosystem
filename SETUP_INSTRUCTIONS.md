# GenAI Telecom Support Chatbot - Setup & Run Instructions

## 📋 Prerequisites

- **Windows OS** (or Mac/Linux with adjustments)
- **Git** installed
- **Python 3.11+** (we recommend Python 3.11 for best compatibility with AI libraries)
- **Internet connection** (for downloading models and dependencies)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Clone the Repository
```bash
git clone https://github.com/ayushikhandal/genai-powered-hybrid-telecom-support-ecosystem.git
cd "genai-powered-hybrid-telecom-support-ecosystem"
```

### Step 2: Create a Virtual Environment (Python 3.11)
```bash
# Create virtual environment with Python 3.11
py -3.11 -m venv venv311

# Activate it
# On Windows:
venv311\Scripts\activate.bat
# On Mac/Linux:
source venv311/bin/activate
```

**Note:** If you don't have Python 3.11, install it:
- **Windows:** `winget install Python.Python.3.11`
- **Mac:** `brew install python@3.11`
- **Linux:** `sudo apt-get install python3.11 python3.11-venv`

### Step 3: Install Dependencies & Run
```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version - smaller, faster)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install TensorFlow and Keras compatibility
pip install tensorflow tf-keras

# Install project dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run genai_telecom_support/chatbot_app.py
```

The app will open at: **http://localhost:8501**

---

## 🐛 Troubleshooting

### Issue: "Python 3.11 not found"
**Solution:** Install Python 3.11 from python.org or use:
```bash
winget install Python.Python.3.11  # Windows
brew install python@3.11            # Mac
```

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:** Make sure you're in the virtual environment:
```bash
# Windows
venv311\Scripts\activate.bat
# Mac/Linux
source venv311/bin/activate
```

Then reinstall:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "LLM backend not available"
**Solution:** This means PyTorch/TensorFlow didn't load. Check:
1. You're using Python 3.11+ (not 3.13)
2. Virtual environment is activated
3. All packages installed: `pip list | grep -E "torch|tensorflow"`

### Issue: "Keras 3 not supported"
**Solution:** Install tf-keras:
```bash
pip install tf-keras
```

### Issue: Port 8501 already in use
**Solution:** Run on a different port:
```bash
streamlit run genai_telecom_support/chatbot_app.py --server.port 8502
```

---

## 📊 Project Structure

```
genai-powered-hybrid-telecom-support-ecosystem/
├── genai_telecom_support/
│   ├── chatbot_app.py          # Main Streamlit app
│   ├── __init__.py
│   └── escalation_messages.jsonl
├── synthetic_data.json          # Knowledge base
├── generate_kb.py               # KB generator
├── requirements.txt             # Dependencies
├── README.md                    # Project overview
├── START_HERE.md                # Quick start guide
└── DEMO_*.md                    # Demo documentation
```

---

## 🎯 Features Overview

| Feature | Status | Notes |
|---------|--------|-------|
| Knowledge Base (KB) Retrieval | ✅ | 40+ Q&A entries |
| Sentiment Analysis | ✅ | Detects frustrated/neutral/positive |
| Confidence Scoring | ✅ | Multi-factor scoring (0-95%) |
| Real-time Escalation | ✅ | Escalates to human agents |
| LLM Integration | ✅ | Uses PyTorch + TensorFlow |
| Step-by-Step Guidance | ✅ | Emoji-based instructions |
| Feedback Loop | ✅ | Learns from escalations |
| Analytics Dashboard | ✅ | Tracks metrics & KPIs |
| Agent Dashboard | ✅ | Real-time escalation management |

---

## 🧪 Testing the App

### Test 1: KB Resolution
**Question:** "My battery is draining fast"
**Expected:** KB answer with 75% confidence + step-by-step guidance

### Test 2: Sentiment Detection
**Question:** "I'm really frustrated with this service!"
**Expected:** Empathetic response + auto-escalation option

### Test 3: LLM Resolution
**Question:** "Explain how 5G networks work"
**Expected:** Custom LLM-generated answer (if LLM enabled)

### Test 4: Escalation
**Question:** "Something weird is happening"
**Expected:** Low confidence → escalation to human agent option

---

## 📝 System Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| Python | 3.11+ | 3.13 has compatibility issues |
| RAM | 8GB minimum | 16GB recommended for LLM |
| Disk | 5GB minimum | For dependencies + models |
| Internet | Required | For model downloads |
| OS | Windows/Mac/Linux | Tested on Windows 11 |

---

## 🔧 Advanced Setup (For Developers)

### Using Conda instead of venv
```bash
conda create -n telecom python=3.11
conda activate telecom
pip install -r requirements.txt
streamlit run genai_telecom_support/chatbot_app.py
```

### GPU Support (NVIDIA)
To use GPU instead of CPU:
```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Environment Variables
```bash
# Disable TensorFlow warnings (optional)
set TF_ENABLE_ONEDNN_OPTS=0

# Use Hugging Face API (if local models don't work)
set HUGGINGFACE_API_KEY=your_token_here

# Disable Hugging Face symlink warning
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

---

## 📚 Project Files Guide

| File | Purpose |
|------|---------|
| `chatbot_app.py` | Main Streamlit application |
| `synthetic_data.json` | Knowledge base with 40+ Q&A pairs |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview & features |
| `START_HERE.md` | Quick start guide |
| `DEMO_HIGHLIGHTS.md` | Demo talking points |
| `feedback_kb.jsonl` | Stores user feedback |
| `escalation_messages.jsonl` | Stores escalation messages |

---

## 🎓 Key Technologies

- **Frontend:** Streamlit (real-time UI)
- **NLP:** TextBlob (sentiment analysis), Transformers (LLM)
- **ML:** PyTorch, TensorFlow, Keras
- **Data:** JSONL (scalable storage)
- **Language:** Python 3.11

---

## 📞 Support

If you encounter issues:

1. **Check Python version:** `python --version`
2. **Check virtual environment:** `pip list | grep streamlit`
3. **Check LLM support:** Run `python -c "import torch; import tensorflow; print('OK')"`
4. **Re-install dependencies:** `pip install --force-reinstall -r requirements.txt`

---

## ✅ Verification Checklist

Before running:
- [ ] Python 3.11+ installed
- [ ] Virtual environment created and activated
- [ ] `pip list` shows streamlit, torch, tensorflow, transformers
- [ ] requirements.txt in project root
- [ ] `genai_telecom_support/chatbot_app.py` exists

After starting:
- [ ] Browser opens to http://localhost:8501
- [ ] No errors in terminal
- [ ] Chatbot interface loads
- [ ] You can type questions

---

## 🚀 Deployment (Optional)

To deploy on **Streamlit Cloud** for judges:

1. Push to GitHub: `git push origin main`
2. Go to: https://streamlit.io/cloud
3. Click "New app" → Select your repo
4. Set main file: `genai_telecom_support/chatbot_app.py`
5. Click "Deploy"

Streamlit Cloud will automatically install dependencies from `requirements.txt`!

---

**Ready to test? Follow the 3 steps above and start chatting! 🎉**
