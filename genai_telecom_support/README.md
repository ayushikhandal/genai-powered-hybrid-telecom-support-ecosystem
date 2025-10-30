GenAI-Powered Hybrid Telecom Support Chatbot

Quickstart

1. Create a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. (Optional) Install torch if you want local LLMs:

```powershell
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

3. Run the Streamlit app:

```powershell
python -m streamlit run chatbot_app.py
```

Notes
- If you don't install torch locally, the app will fall back to KB-only responses. You can also set `HUGGINGFACE_API_KEY` to use the Hugging Face Inference API for dynamic LLM responses.
- Edit `providers.json` in the app folder or use the Admin sidebar to manage provider name variants and typos.
