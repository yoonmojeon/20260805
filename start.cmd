@echo off
setlocal
cd /d "%~dp0"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "PYTHONUTF8=1"

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] 이 폴더에 새 가상환경 생성 중...
  py -3 -m venv .venv
  if errorlevel 1 (
    echo py launcher 실패. python -m venv 로 재시도합니다.
    python -m venv .venv
  )
  echo [2/3] 실행용 패키지 설치 중. 몇 분 걸릴 수 있습니다.
  ".venv\Scripts\python.exe" -m pip install -U pip
  ".venv\Scripts\python.exe" -m pip install markdown-it-py openai "pandas==2.2.3" numpy python-docx folium "gradio>=4.36.0" openpyxl requests chromadb sentence-transformers rank_bm25 PyMuPDF Pillow tqdm "grpcio==1.62.3"
)

echo [3/3] 앱 시작: http://127.0.0.1:7860
".venv\Scripts\python.exe" app.py
pause
