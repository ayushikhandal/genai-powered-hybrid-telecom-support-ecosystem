# ⚡ Quick Run Commands for Judges

Copy & paste these commands to run the project immediately:

## Windows (Recommended)

```powershell
# 1. Clone the repo
git clone https://github.com/ayushikhandal/genai-powered-hybrid-telecom-support-ecosystem.git
cd genai-powered-hybrid-telecom-support-ecosystem

# 2. Create Python 3.11 virtual environment
py -3.11 -m venv venv311

# 3. Activate virtual environment
venv311\Scripts\activate.bat

# 4. Install dependencies (this takes 3-5 minutes)
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow tf-keras
pip install -r requirements.txt

# 5. Run the app
streamlit run genai_telecom_support/chatbot_app.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Open **http://localhost:8501** in your browser! 🎉

---

## Mac/Linux

```bash
# 1. Clone the repo
git clone https://github.com/ayushikhandal/genai-powered-hybrid-telecom-support-ecosystem.git
cd genai-powered-hybrid-telecom-support-ecosystem

# 2. Create Python 3.11 virtual environment
python3.11 -m venv venv311

# 3. Activate virtual environment
source venv311/bin/activate

# 4. Install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow tf-keras
pip install -r requirements.txt

# 5. Run the app
streamlit run genai_telecom_support/chatbot_app.py
```

---

## ⏱️ Installation Time

- **Python 3.11 setup:** ~2 minutes
- **PyTorch install:** ~2 minutes
- **TensorFlow install:** ~3 minutes
- **Other dependencies:** ~1 minute
- **Total:** ~8-10 minutes (first time only)

---

## 🧪 Test Questions to Try

1. **KB Test:** "My battery is draining fast"
   - Expected: Step-by-step solution with 75% confidence

2. **Sentiment Test:** "I'm so frustrated with this!"
   - Expected: Empathetic tone + escalation option

3. **LLM Test:** "Explain how 5G networks work"
   - Expected: Custom AI-generated answer

4. **Escalation Test:** "Something weird is happening"
   - Expected: Low confidence + escalation button

---

## 📊 Dashboard Metrics

Look for:
- ✅ **KB resolved:** Queries answered from knowledge base
- ✅ **LLM resolved:** Queries answered by AI model
- ✅ **Escalations:** Queries sent to human agents
- ✅ **Confidence:** Ranging from 25% to 95%
- ✅ **Sentiment:** Frustrated, Neutral, or Positive

---

## ⚠️ If You Get Errors

### "Python 3.11 not found"
```bash
# Install Python 3.11
winget install Python.Python.3.11  # Windows
brew install python@3.11            # Mac
```

### "ModuleNotFoundError: torch"
```bash
# Make sure venv is activated, then reinstall
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### "LLM backend not available"
```bash
# Install tf-keras to fix Keras compatibility
pip install tf-keras
```

### Port 8501 already in use
```bash
# Use different port
streamlit run genai_telecom_support/chatbot_app.py --server.port 8502
```

---

## 📁 Project Structure

```
genai-powered-hybrid-telecom-support-ecosystem/
├── genai_telecom_support/
│   └── chatbot_app.py           ← Main app (this is what runs!)
├── synthetic_data.json          ← Knowledge base (40+ Q&A)
├── requirements.txt             ← Dependencies (install these)
├── SETUP_INSTRUCTIONS.md        ← Detailed setup guide
└── README.md                    ← Full documentation
```

---

## ✨ What You'll See

1. **Chat Interface** - Ask telecom questions
2. **Confidence Score** - Shows how confident the AI is (25%-95%)
3. **Sentiment Detection** - AI detects if you're frustrated
4. **Step-by-Step Guidance** - Emoji-based instructions
5. **Escalation Option** - For complex issues
6. **Dashboard** - Analytics and metrics
7. **Agent Dashboard** - Escalation management

---

## 🎯 Key Features to Highlight

| Feature | Why It's Cool |
|---------|--------------|
| **Hybrid AI** | Uses KB + LLM together (best of both) |
| **Sentiment Analysis** | Understands customer emotion |
| **Real-time Escalation** | Seamless handoff to humans |
| **Confidence Scoring** | Shows how sure the AI is |
| **Self-Learning** | Improves from escalations |
| **Multi-language UI** | Clean, professional interface |

---

## 🚀 Next Steps After Running

1. **Ask a question** - Try "my sim is not working"
2. **Check confidence** - Notice the 75% confidence score
3. **Read guidance** - See the emoji-based steps
4. **Test sentiment** - Try frustrated tone in question
5. **Explore escalation** - Click "Escalate" button
6. **Check dashboard** - View metrics and analytics

---

**Enjoy! The app should work out of the box! 🎉**

Questions? Check SETUP_INSTRUCTIONS.md for detailed help.
