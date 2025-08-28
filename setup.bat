@echo off
echo 📚 Iniciando el Agente de Recomendacion de Libros
echo.

echo Instalando dependencias de la UI...
cd ui
pip install -r requirements.txt
cd ..

echo.
echo Instalando dependencias principales...
pip install -r requirements.txt

echo.
echo ✅ Dependencias instaladas.
echo.
echo Para usar el sistema:
echo 1. Ejecuta 'start_api.py' para iniciar la API
echo 2. Ejecuta 'streamlit run ui/streamlit_app.py' para la interfaz web
echo.
echo O usa los siguientes comandos:
echo   python start_api.py
echo   streamlit run ui/streamlit_app.py
echo.
pause
