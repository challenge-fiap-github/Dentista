#!/usr/bin/env python3
"""
Script de Instalação Automática do OdontoVision
Execute: python setup.py
"""

import os
import sys
import subprocess
import platform

def print_header():
    """Imprime cabeçalho bonito"""
    print("\n" + "="*60)
    print("🦷 OdontoVision - Setup Automático")
    print("="*60 + "\n")

def check_python_version():
    """Verifica versão do Python"""
    print("📋 Verificando versão do Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ é necessário!")
        print(f"   Versão atual: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detectado\n")

def install_dependencies():
    """Instala dependências Python"""
    print("📦 Instalando dependências Python...")
    
    packages = [
        'flask',
        'flask-cors',
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib'
    ]
    
    try:
        for package in packages:
            print(f"   Instalando {package}...")
            subprocess.check_call([
                sys.executable, 
                '-m', 
                'pip', 
                'install', 
                package,
                '--quiet'
            ])
        print("✅ Todas as dependências instaladas!\n")
        return True
    except Exception as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

def create_directory_structure():
    """Cria estrutura de diretórios"""
    print("📁 Criando estrutura de diretórios...")
    
    dirs = [
        'backend',
        'frontend',
        'frontend/admin',
        'data',
        'models'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"   ✓ {dir_path}")
    
    print("✅ Estrutura de diretórios criada!\n")

def copy_files():
    """Copia arquivos para os diretórios corretos"""
    print("📄 Copiando arquivos...")
    
    files = {
        'app.py': 'backend/',
        'gerar_dados_realistas.py': 'backend/',
        'admin-dashboard-melhorado.html': 'frontend/admin/dashboard.html',
        '.env.example': '.env.example'
    }
    
    for src, dest in files.items():
        if os.path.exists(src):
            dest_path = os.path.join(os.getcwd(), dest)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Copiar arquivo
            import shutil
            shutil.copy2(src, dest_path)
            print(f"   ✓ {src} → {dest}")
    
    print("✅ Arquivos copiados!\n")

def generate_data():
    """Gera dados realistas"""
    print("🔄 Gerando dados realistas...")
    print("   (Isso pode levar alguns segundos...)\n")
    
    try:
        # Executar gerador de dados
        script_path = os.path.join('backend', 'gerar_dados_realistas.py')
        if os.path.exists(script_path):
            subprocess.check_call([sys.executable, script_path])
            print("\n✅ Dados gerados com sucesso!\n")
            return True
        else:
            print("⚠️ Script gerador não encontrado")
            return False
    except Exception as e:
        print(f"❌ Erro ao gerar dados: {e}")
        return False

def create_gitignore():
    """Cria arquivo .gitignore"""
    print("📝 Criando .gitignore...")
    
    gitignore_content = """# OdontoVision - Git Ignore

# Variáveis de ambiente
.env
.env.local
.env.production
*.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Dados e Modelos (opcional - descomentar se quiser versionar)
# data/*.csv
# models/*.pkl
# models/*.json

# Logs
*.log
logs/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore criado!\n")

def print_next_steps():
    """Imprime próximos passos"""
    print("\n" + "="*60)
    print("✨ Instalação concluída com sucesso!")
    print("="*60 + "\n")
    
    print("📋 PRÓXIMOS PASSOS:\n")
    
    print("1️⃣  Configurar Email (Opcional)")
    print("   $ cp .env.example .env")
    print("   $ nano .env  # Edite com suas credenciais")
    print("")
    
    print("2️⃣  Iniciar Backend")
    print("   $ cd backend")
    print("   $ python app.py")
    print("")
    
    print("3️⃣  Acessar Frontend")
    print("   Abra no navegador: frontend/admin/dashboard.html")
    print("")
    
    print("4️⃣  Fazer Login")
    print("   Usuário: admin")
    print("   Senha: admin123")
    print("")
    
    print("="*60)
    print("📚 Documentação:")
    print("   - README.md - Guia rápido")
    print("   - GUIA_COMPLETO_SISTEMA.md - Guia detalhado")
    print("="*60 + "\n")

def main():
    """Função principal"""
    try:
        print_header()
        check_python_version()
        
        # Confirmar instalação
        print("Este script irá:")
        print("  • Instalar dependências Python")
        print("  • Criar estrutura de diretórios")
        print("  • Gerar dados de exemplo")
        print("  • Configurar o projeto\n")
        
        response = input("Deseja continuar? (s/n): ").lower()
        if response != 's':
            print("\n❌ Instalação cancelada pelo usuário")
            sys.exit(0)
        
        print()
        
        # Executar instalação
        if not install_dependencies():
            print("\n⚠️ Erro ao instalar dependências. Continuando...")
        
        create_directory_structure()
        # copy_files()  # Comentado pois os arquivos já estão em outputs
        create_gitignore()
        
        # Gerar dados
        if os.path.exists(os.path.join('backend', 'gerar_dados_realistas.py')):
            generate_data()
        else:
            print("⚠️ Coloque os arquivos Python em backend/ antes de gerar dados\n")
        
        print_next_steps()
        
        print("🎉 Instalação concluída! Bom uso!\n")
        
    except KeyboardInterrupt:
        print("\n\n❌ Instalação interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro durante instalação: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()