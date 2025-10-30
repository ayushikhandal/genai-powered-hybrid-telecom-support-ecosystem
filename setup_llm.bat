@echo off
cd "c:\Users\khandal\OneDrive - Nokia\Desktop\Strong Her hackthon"

REM Activate virtual environment
call venv311\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip setuptools wheel

REM Install PyTorch CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install TensorFlow
pip install tensorflow

REM Install other dependencies
pip install streamlit textblob transformers requests

REM Run the app
echo.
echo Activation complete! Now running Streamlit app...
echo.
streamlit run genai_telecom_support\chatbot_app.py

pause
