@echo off
REM Force Python 3.11 virtual environment
cd /d "c:\Users\khandal\OneDrive - Nokia\Desktop\Strong Her hackthon"

REM Activate the Python 3.11 venv
call venv311\Scripts\activate.bat

REM Verify Python version
echo Checking Python version...
python --version

REM Run Streamlit
echo Starting Streamlit app with LLM support...
streamlit run genai_telecom_support\chatbot_app.py

pause
