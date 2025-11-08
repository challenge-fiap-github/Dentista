#!/bin/bash

echo "🦷 OdontoVision - Sistema de Detecção de Fraudes Odontológicas"
echo "================================================================"
echo ""

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Verificar se está no diretório correto
if [ ! -f "backend/app.py" ]; then
    echo "❌ Execute este script do diretório OdontoVision/"
    exit 1
fi

# Instalar dependências
echo -e "${BLUE}📦 Instalando dependências...${NC}"
pip install --break-system-packages -r backend/requirements.txt --quiet

echo ""
echo -e "${GREEN}✅ Dependências instaladas!${NC}"
echo ""

# Verificar se os dados existem
if [ ! -f "data/odonto_vision_data.csv" ]; then
    echo -e "${BLUE}📊 Gerando dados realistas...${NC}"
    python backend/gerar_dados_realistas.py
    echo ""
fi

echo -e "${GREEN}✅ Sistema pronto para iniciar!${NC}"
echo ""
echo "================================================"
echo "📝 INFORMAÇÕES DE ACESSO:"
echo "================================================"
echo ""
echo "🌐 Backend API: http://localhost:5000"
echo "🖥️  Frontend:   Abra os arquivos HTML no navegador"
echo ""
echo "👤 USUÁRIOS DE DEMONSTRAÇÃO:"
echo "   Admin:     admin / admin123"
echo "   Dentista:  dentista1 / dent123"
echo ""
echo "================================================"
echo ""
echo -e "${BLUE}🚀 Iniciando servidor backend...${NC}"
echo ""

# Iniciar backend
cd backend
python app.py
