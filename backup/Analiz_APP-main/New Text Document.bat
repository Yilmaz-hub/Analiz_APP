@echo off
echo Kripto Analiz Programi Baslatiliyor...
cd /d "%~dp0"
python -m streamlit run app.py
pause