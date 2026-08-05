@echo off
cd /d C:\projects\MaritimeRAG
"C:\Users\rlaeh\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run scripts\15_rag_ui.py --server.port 8501 --server.headless true
