#!/bin/bash
# Script de entrada para Docker

set -e

# Función para iniciar la API
start_api() {
    echo "🚀 Iniciando API FastAPI..."
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &
}

# Función para iniciar Streamlit
start_streamlit() {
    echo "🖥️ Iniciando interfaz Streamlit..."
    streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
}

# Verificar variables de entorno obligatorias
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY no está configurada"
    echo "Por favor, configura tu API key de OpenAI"
    exit 1
fi

# Crear directorio de base de datos si no existe
mkdir -p /app/data

# Inicializar la base de datos si es necesario
python -c "
from app.api.database import init_database
init_database()
print('✅ Base de datos inicializada')
"

echo "📚 Book Recommender AI Agent - Iniciando servicios..."

# Determinar qué servicios iniciar basado en argumentos
case "${1:-all}" in
    "api")
        start_api
        wait
        ;;
    "streamlit")
        start_streamlit
        wait
        ;;
    "all"|*)
        start_api
        sleep 2
        start_streamlit
        wait
        ;;
esac
