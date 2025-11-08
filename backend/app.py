"""
Backend OdontoVision - Sistema Inteligente de Detecção de Fraudes Odontológicas
Com Machine Learning que aprende a cada classificação
VERSÃO ATUALIZADA com endpoint de email e múltiplos status
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime, timedelta
import json
import os
import hashlib
from functools import wraps
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
CORS(app)

# Desabilita cache para todas as respostas
@app.after_request
def disable_cache(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# ==================== CONFIGURAÇÕES ====================
# Caminhos relativos que funcionam em Windows, Mac e Linux
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE = os.path.join(BASE_DIR, 'data', 'odonto_vision_data.csv')
MODEL_FILE = os.path.join(BASE_DIR, 'models', 'fraud_model.pkl')
ENCODER_FILE = os.path.join(BASE_DIR, 'models', 'label_encoders.pkl')
METRICS_FILE = os.path.join(BASE_DIR, 'models', 'model_metrics.json')

# Configurações de Email (usar variáveis de ambiente em produção)
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USER = os.getenv('SMTP_USER', 'odontovision@example.com')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
EMAIL_FROM = os.getenv('EMAIL_FROM', 'OdontoVision <noreply@odontovision.com.br>')

# Usuários do sistema (em produção usar banco de dados)
USERS = {
    'admin': {
        'password': hashlib.sha256('admin123'.encode()).hexdigest(),
        'role': 'admin',
        'name': 'Administrador do Sistema',
        'email': 'admin@odontovision.com.br'
    },
    'dentista1': {
        'password': hashlib.sha256('dent123'.encode()).hexdigest(),
        'role': 'dentista',
        'name': 'Dr. Marcos Antonio Silveira',
        'cro': 'CRO-SP 45821',
        'email': 'marcos.silveira@odontovision.com.br'
    },
    'dentista2': {
        'password': hashlib.sha256('dent123'.encode()).hexdigest(),
        'role': 'dentista',
        'name': 'Dra. Patricia Mendes Costa',
        'cro': 'CRO-RJ 32156',
        'email': 'patricia.costa@odontovision.com.br'
    }
}

# ==================== AUTENTICAÇÃO ====================
def require_auth(role=None):
    """Decorator para verificar autenticação"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return jsonify({'error': 'Autenticação necessária'}), 401
            
            try:
                # Bearer token format
                token = auth_header.split(' ')[1] if ' ' in auth_header else auth_header
                # Em produção, validar JWT aqui
                username = token.split(':')[0]  # Simplificado
                
                if username not in USERS:
                    return jsonify({'error': 'Usuário inválido'}), 401
                
                if role and USERS[username]['role'] != role:
                    return jsonify({'error': 'Permissão negada'}), 403
                
                # Adicionar usuário ao request
                request.user = USERS[username]
                request.username = username
                
            except Exception as e:
                return jsonify({'error': 'Token inválido'}), 401
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ==================== SERVIÇO DE EMAIL ====================
def enviar_email(destinatario, assunto, corpo_html):
    """
    Envia email usando SMTP
    Retorna: {'sucesso': bool, 'erro': str ou None}
    """
    try:
        # Se não houver configuração de email, simular envio
        if not SMTP_PASSWORD:
            print(f"📧 [SIMULAÇÃO] Email enviado para: {destinatario}")
            print(f"   Assunto: {assunto}")
            return {'sucesso': True, 'erro': None}
        
        # Criar mensagem
        mensagem = MIMEMultipart('alternative')
        mensagem['Subject'] = assunto
        mensagem['From'] = EMAIL_FROM
        mensagem['To'] = destinatario
        
        # Adicionar corpo HTML
        parte_html = MIMEText(corpo_html, 'html', 'utf-8')
        mensagem.attach(parte_html)
        
        # Conectar ao servidor SMTP e enviar
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(mensagem)
        
        print(f"✅ Email enviado para: {destinatario}")
        return {'sucesso': True, 'erro': None}
    
    except Exception as e:
        print(f"❌ Erro ao enviar email: {str(e)}")
        return {'sucesso': False, 'erro': str(e)}

# ==================== CARREGAMENTO DE DADOS ====================
class OdontoVisionML:
    """Sistema de ML que aprende com classificações"""
    
    def __init__(self):
        self.df = None
        self.model = None
        self.encoders = {}
        self.feature_columns = []
        self.metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'total_predictions': 0,
            'correct_predictions': 0,
            'last_trained': None,
            'training_history': []
        }
        self.load_data()
        self.load_or_train_model()
    
    def load_data(self):
        """Carrega dados do CSV"""
        try:
            if not os.path.exists(CSV_FILE):
                print(f"⚠️ Arquivo CSV não encontrado em: {CSV_FILE}")
                print(f"📂 Gerando dados agora...")
                # Tentar gerar os dados
                try:
                    import subprocess
                    script_path = os.path.join(os.path.dirname(__file__), 'gerar_dados_realistas.py')
                    subprocess.run(['python', script_path], check=True)
                    print("✅ Dados gerados com sucesso!")
                except Exception as e:
                    print(f"❌ Erro ao gerar dados: {e}")
                    print(f"💡 Execute manualmente: python backend/gerar_dados_realistas.py")
                    self.df = pd.DataFrame()
                    return
            
            self.df = pd.read_csv(CSV_FILE)
            
            # Garantir que CPF seja string
            if 'cpf' in self.df.columns:
                self.df['cpf'] = self.df['cpf'].astype(str)
            
            # Garantir formato correto de datas
            if 'created_at' in self.df.columns:
                self.df['created_at'] = pd.to_datetime(self.df['created_at'], format='ISO8601', errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Garantir que a coluna dentista_email existe
            if 'dentista_email' not in self.df.columns:
                print("⚠️ Adicionando coluna dentista_email aos dados...")
                self.df['dentista_email'] = self.df['dentista'].apply(
                    lambda x: x.lower().replace(' ', '.').replace('dr.', '').replace('dra.', '').strip() + '@odontovision.com.br'
                )
                self.df.to_csv(CSV_FILE, index=False)
            
            print(f"✅ Dados carregados: {len(self.df)} registros")
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            self.df = pd.DataFrame()
    
    def prepare_features(self, df):
        """Prepara features para o modelo"""
        df = df.copy()
        
        # Converter data para features temporais
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['mes'] = df['created_at'].dt.month
        df['dia_semana'] = df['created_at'].dt.dayofweek
        df['ano'] = df['created_at'].dt.year
        
        # Features categóricas
        categorical_cols = ['procedimento', 'dentista']
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(df[col].astype(str))
            
            try:
                df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
            except:
                # Lidar com novos valores
                df[f'{col}_encoded'] = 0
        
        # Features numéricas
        numeric_features = ['mes', 'dia_semana', 'ano', 'dente']
        encoded_features = [f'{col}_encoded' for col in categorical_cols]
        
        self.feature_columns = numeric_features + encoded_features
        
        return df[self.feature_columns].fillna(0)
    
    def train_model(self, force=False):
        """Treina ou retreina o modelo"""
        if len(self.df) == 0:
            print("⚠️ Nenhum dado disponível para treinar modelo")
            return
        
        if 'fraude' not in self.df.columns:
            print("⚠️ Coluna 'fraude' não encontrada nos dados")
            return
            
        if len(self.df[self.df['fraude'].notna()]) < 50:
            print("⚠️ Dados insuficientes para treinar modelo")
            return
        
        print("🤖 Treinando modelo de Machine Learning...")
        
        # Preparar dados
        df_train = self.df[self.df['fraude'].notna()].copy()
        X = self.prepare_features(df_train)
        y = df_train['fraude'].astype(int)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Treinar modelo (usando Gradient Boosting para melhor performance)
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Avaliar
        y_pred = self.model.predict(X_test)
        
        self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
        self.metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
        self.metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
        self.metrics['f1_score'] = f1_score(y_test, y_pred, zero_division=0)
        self.metrics['last_trained'] = datetime.now().isoformat()
        
        # Histórico
        self.metrics['training_history'].append({
            'timestamp': datetime.now().isoformat(),
            'accuracy': self.metrics['accuracy'],
            'precision': self.metrics['precision'],
            'recall': self.metrics['recall'],
            'f1_score': self.metrics['f1_score'],
            'training_samples': len(X_train)
        })
        
        # Manter apenas últimos 50 treinos
        self.metrics['training_history'] = self.metrics['training_history'][-50:]
        
        # Salvar
        self.save_model()
        
        print(f"✅ Modelo treinado!")
        print(f"   Acurácia: {self.metrics['accuracy']:.2%}")
        print(f"   Precisão: {self.metrics['precision']:.2%}")
        print(f"   Recall: {self.metrics['recall']:.2%}")
        print(f"   F1-Score: {self.metrics['f1_score']:.2%}")
    
    def load_or_train_model(self):
        """Carrega modelo existente ou treina novo"""
        try:
            if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
                self.model = joblib.load(MODEL_FILE)
                self.encoders = joblib.load(ENCODER_FILE)
                if os.path.exists(METRICS_FILE):
                    with open(METRICS_FILE, 'r') as f:
                        self.metrics = json.load(f)
                print("✅ Modelo carregado do disco")
            else:
                self.train_model()
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo: {e}")
            self.train_model()
    
    def save_model(self):
        """Salva modelo e métricas"""
        try:
            os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
            joblib.dump(self.model, MODEL_FILE)
            joblib.dump(self.encoders, ENCODER_FILE)
            with open(METRICS_FILE, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            print("💾 Modelo salvo")
        except Exception as e:
            print(f"❌ Erro ao salvar modelo: {e}")

    def reload_data(self):
        """Recarrega dados do CSV para pegar atualizações"""
        try:
            if os.path.exists(CSV_FILE):
                self.df = pd.read_csv(CSV_FILE)
                
                # Garantir tipos corretos
                if 'id' in self.df.columns:
                    self.df['id'] = pd.to_numeric(self.df['id'], errors='coerce')
                
                if 'cpf' in self.df.columns:
                    self.df['cpf'] = self.df['cpf'].astype(str)
                
                if 'created_at' in self.df.columns:
                    # manter como string ISO; parsing quando necessário
                    pass
                
                print(f"📂 Dados recarregados: {len(self.df)} registros")
                return True
        except Exception as e:
            print(f"❌ Erro ao recarregar dados: {e}")
            return False
    
    def predict(self, registro):
        """Faz previsão para um registro"""
        if self.model is None:
            return {'fraude_detectada': False, 'confianca': 0, 'motivos': []}
        
        try:
            # Preparar features
            df_temp = pd.DataFrame([registro])
            X = self.prepare_features(df_temp)
            
            # Predição
            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            
            return {
                'fraude_detectada': bool(pred),
                'confianca': float(proba[1]),
                'probabilidade_fraude': float(proba[1] * 100)
            }
        except Exception as e:
            print(f"Erro na predição: {e}")
            return {'fraude_detectada': False, 'confianca': 0, 'motivos': []}
    
    def learn_from_classification(self, registro_id, classificacao_admin):
        """
        Aprende com a classificação do admin
        Retreina o modelo quando houver classificações suficientes
        """
        try:
            # Atualizar label no dataset
            idx = self.df[self.df['id'] == registro_id].index
            if len(idx) > 0:
                self.df.loc[idx[0], 'fraude'] = 1 if classificacao_admin == 'fraude' else 0
                self.df.to_csv(CSV_FILE, index=False)
                
                # Retreinar a cada 10 classificações
                classificacoes_recentes = len(self.metrics.get('training_history', []))
                if classificacoes_recentes % 10 == 0:
                    print("🔄 Retreinando modelo com novas classificações...")
                    self.train_model()
                    return True
        except Exception as e:
            print(f"Erro ao aprender: {e}")
        
        return False

# Inicializar sistema
ml_system = OdontoVisionML()

# ==================== ROTAS DE AUTENTICAÇÃO ====================

@app.route('/api/login', methods=['POST'])
def login():
    """Login de usuário"""
    data = request.json
    username = data.get('username', '').lower()
    password = data.get('password', '')
    
    if username not in USERS:
        return jsonify({'error': 'Usuário ou senha inválidos'}), 401
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    if USERS[username]['password'] != password_hash:
        return jsonify({'error': 'Usuário ou senha inválidos'}), 401
    
    # Em produção, gerar JWT aqui
    token = f"{username}:token"
    
    return jsonify({
        'success': True,
        'token': token,
        'user': {
            'username': username,
            'role': USERS[username]['role'],
            'name': USERS[username]['name']
        }
    })

# ==================== ROTAS DO DENTISTA ====================

@app.route('/api/dentista/consultas', methods=['POST'])
@require_auth(role='dentista')
def criar_consulta():
    """Dentista registra nova consulta"""
    try:
        data = request.json
        
        # Validação
        required_fields = ['paciente_nome', 'cpf', 'procedimento', 'executado', 'dente']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Campo obrigatório: {field}'}), 400

        # Gerar novo ID
        if ml_system.df is None:
            ml_system.df = pd.DataFrame()

        if len(ml_system.df) > 0 and 'id' in ml_system.df.columns:
            # Garantir que IDs são numéricos
            ml_system.df['id'] = pd.to_numeric(ml_system.df['id'], errors='coerce')
            new_id = int(ml_system.df['id'].max()) + 1 if not ml_system.df['id'].isna().all() else 1
        else:
            new_id = 1

        # Preparar registro
        registro = {
            'id': new_id,
            'paciente_nome': data['paciente_nome'],
            'cpf': str(data['cpf']).replace('.', '').replace('-', ''),
            'procedimento': data['procedimento'],
            'codigo_tuss': data.get('codigo_tuss', ''),
            'executado': data['executado'],
            'prescricao': data.get('prescricao', 'Sem prescrição'),
            'dente': int(data['dente']),
            'dentista': request.user['name'],
            'dentista_email': request.user.get('email', ''),
            'cro_dentista': request.user.get('cro', ''),
            'created_at': datetime.now().isoformat(),
            'fraude': None,  # Será determinado pelo admin
            'status': 'novo',
            'motivo_atencao': '',
            'suspicious': False
        }

        # Fazer predição com ML
        predicao = ml_system.predict(registro)

        # Se ML detectar possível fraude, marcar para análise
        if predicao.get('fraude_detectada') and predicao.get('confianca', 0) > 0.7:
            registro['status'] = 'em_analise'
            registro['suspicious'] = True
            registro['motivo_atencao'] = f"ML detectou risco ({predicao['probabilidade_fraude']:.1f}% de probabilidade)"
        elif predicao.get('fraude_detectada') and predicao.get('confianca', 0) > 0.5:
            registro['status'] = 'atencao'
            registro['suspicious'] = True
            registro['motivo_atencao'] = f"ML detectou possível risco ({predicao['probabilidade_fraude']:.1f}% de probabilidade)"

        # Adicionar ao dataframe
        novo_df = pd.DataFrame([registro])
        ml_system.df = pd.concat([ml_system.df, novo_df], ignore_index=True)

        # Salvar no CSV
        ml_system.df.to_csv(CSV_FILE, index=False)
        # 🔄 Opcional: alinhar tipos e refletir imediatamente no próximo GET
        ml_system.reload_data()
        print(f"✅ Nova consulta registrada: ID {new_id} - Status: {registro['status']}")

        return jsonify({
            'success': True,
            'consulta': registro,
            'predicao_ml': predicao
        }), 201
    except Exception as e:
        print(f"❌ Erro ao criar consulta: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erro ao processar: {str(e)}'}), 500

@app.route('/api/dentista/consultas', methods=['GET'])
@require_auth(role='dentista')
def listar_consultas_dentista():
    """Lista consultas do dentista logado (sempre refletindo o CSV mais recente)"""
    try:
        # 🔄 Garante que lemos o CSV atualizado (alterado pelo POST, admin, etc.)
        if hasattr(ml_system, 'reload_data'):
            ml_system.reload_data()
        else:
            ml_system.load_data()

        dentista_nome = request.user['name']

        # Se não há dados
        if ml_system.df is None or ml_system.df.empty:
            return jsonify({'consultas': [], 'total': 0, 'page': 0, 'totalPages': 0})

        # Filtra pelas consultas do dentista
        df_dentista = ml_system.df[ml_system.df['dentista'] == dentista_nome].copy()

        # Ordena por data (mais recentes primeiro) de forma robusta
        if 'created_at' in df_dentista.columns:
            df_dentista['created_at_parsed'] = pd.to_datetime(
                df_dentista['created_at'], errors='coerce'
            )
            df_dentista = df_dentista.sort_values(
                'created_at_parsed', ascending=False, na_position='last'
            ).drop(columns=['created_at_parsed'], errors='ignore')

        # Paginação
        page = int(request.args.get('page', 0))
        size = int(request.args.get('size', 20))
        start, end = page * size, page * size + size

        # Converte para dict com limpeza de tipos
        registros = df_dentista.iloc[start:end].to_dict('records')

        def clean_value(value):
            if value is None:
                return None
            if isinstance(value, (float, np.floating)):
                if math.isnan(value) or math.isinf(value): return None
                return float(value)
            if isinstance(value, (np.integer, int)):     return int(value)
            if isinstance(value, (np.bool_, bool)):      return bool(value)
            try:
                import pandas as _pd
                if isinstance(value, _pd.Timestamp):
                    return value.isoformat()
            except Exception:
                pass
            return value

        cleaned = [{k: clean_value(v) for k, v in row.items()} for row in registros]

        return jsonify({
            'consultas': cleaned,
            'total': int(len(df_dentista)),
            'page': page,
            'totalPages': (len(df_dentista) + size - 1) // size if size > 0 else 0
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({
            'error': f'Erro ao carregar consultas: {str(e)}',
            'consultas': [],
            'total': 0,
            'page': 0,
            'totalPages': 0
        }), 500

# ==================== ROTAS DO ADMIN ====================

@app.route('/api/admin/dashboard/stats', methods=['GET'])
@require_auth(role='admin')
def dashboard_stats():
    """Estatísticas gerais para o dashboard"""
    df = ml_system.df
    
    # Filtros de data
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if start_date:
        df = df[pd.to_datetime(df['created_at'], format='ISO8601') >= pd.to_datetime(start_date)]
    if end_date:
        df = df[pd.to_datetime(df['created_at'], format='ISO8601') <= pd.to_datetime(end_date)]
    
    total_registros = len(df)
    
    # Status
    por_status = df['status'].value_counts().to_dict()
    
    # Fraudes (aceitar múltiplos status)
    fraudes_confirmadas = len(df[df['status'].isin(['reprovado', 'fraude'])])
    em_analise = len(df[df['status'].isin(['atencao', 'em_analise', 'pendente'])])
    aprovados = len(df[df['status'].isin(['aprovado', 'legitimo'])])
    
    # Por dentista
    por_dentista = df.groupby('dentista').agg({
        'id': 'count',
        'status': lambda x: (x.isin(['atencao', 'em_analise', 'reprovado', 'pendente', 'fraude'])).sum()
    }).rename(columns={'id': 'total', 'status': 'suspeitos'}).to_dict('index')
    
    # Por procedimento
    por_procedimento = df.groupby('procedimento').agg({
        'id': 'count',
        'status': lambda x: (x.isin(['atencao', 'em_analise', 'reprovado', 'pendente', 'fraude'])).sum()
    }).rename(columns={'id': 'total', 'status': 'suspeitos'}).to_dict('index')
    
    # Timeline (por mês) - CORREÇÃO AQUI
    df_timeline = df.copy()
    # Usar format='ISO8601' para lidar com diferentes formatos de data
    df_timeline['created_at_parsed'] = pd.to_datetime(df_timeline['created_at'], format='ISO8601', errors='coerce')
    # Remover linhas com datas inválidas
    df_timeline = df_timeline.dropna(subset=['created_at_parsed'])
    df_timeline['mes_ano'] = df_timeline['created_at_parsed'].dt.to_period('M')
    
    timeline = df_timeline.groupby('mes_ano').agg({
        'id': 'count',
        'status': lambda x: (x.isin(['atencao', 'em_analise', 'reprovado', 'pendente', 'fraude'])).sum()
    }).reset_index()
    timeline['mes_ano'] = timeline['mes_ano'].astype(str)
    timeline = timeline.rename(columns={'id': 'total', 'status': 'suspeitos'}).to_dict('records')
    
    # Métricas do ML
    ml_metrics = ml_system.metrics
    
    return jsonify({
        'total_registros': total_registros,
        'fraudes_confirmadas': fraudes_confirmadas,
        'em_analise': em_analise,
        'aprovados': aprovados,
        'por_status': por_status,
        'por_dentista': por_dentista,
        'por_procedimento': por_procedimento,
        'timeline': timeline,
        'ml_metrics': ml_metrics
    })


@app.route('/api/admin/alertas', methods=['GET'])
@require_auth(role='admin')
def listar_alertas():
    """
    Lista alertas com suporte a múltiplos status separados por vírgula
    Ex: ?status=pendente,atencao,em_analise
    """
    try:
        # Recarregar dados para garantir que alterações externas sejam refletidas
        if hasattr(ml_system, 'reload_data'):
            ml_system.reload_data()
        else:
            ml_system.load_data()

        df = ml_system.df

        # Verificar se o DataFrame não está vazio
        if df is None or df.empty:
            return jsonify({
                'alertas': [],
                'total': 0,
                'page': 0,
                'totalPages': 0
            })

        # Filtrar por múltiplos status
        status_param = request.args.get('status', '')

        if status_param:
            status_list = [s.strip() for s in status_param.split(',') if s.strip()]
            if status_list:
                if 'status' in df.columns:
                    df_alertas = df[df['status'].isin(status_list)].copy()
                else:
                    df_alertas = df.copy()
            else:
                df_alertas = df.copy()
        else:
            df_alertas = df.copy()

        # Ordenar por data
        if 'created_at' in df_alertas.columns:
            # Tentar ordenar mesmo que existam formatos diferentes
            df_alertas['created_at_parsed'] = pd.to_datetime(df_alertas['created_at'], errors='coerce')
            df_alertas = df_alertas.sort_values('created_at_parsed', ascending=False, na_position='last')
            df_alertas = df_alertas.drop(columns=['created_at_parsed'])

        # Busca
        search = (request.args.get('search') or '').strip().lower()
        if search and len(df_alertas) > 0:
            mask = pd.Series([False] * len(df_alertas), index=df_alertas.index)

            if 'paciente_nome' in df_alertas.columns:
                mask |= df_alertas['paciente_nome'].astype(str).str.lower().str.contains(search, na=False)

            if 'dentista' in df_alertas.columns:
                mask |= df_alertas['dentista'].astype(str).str.lower().str.contains(search, na=False)

            if 'cpf' in df_alertas.columns:
                mask |= df_alertas['cpf'].astype(str).str.contains(search, na=False)

            df_alertas = df_alertas[mask]

        # Paginação
        page = int(request.args.get('page', 0))
        size = int(request.args.get('size', 50))

        start = page * size
        end = start + size

        # Converter para lista de dicionários
        alertas_list = df_alertas.iloc[start:end].to_dict('records')

        # Função para limpar valores NaN/Infinity
        def clean_value(value):
            if value is None:
                return None
            # numpy floats
            if isinstance(value, (float, np.floating)):
                if math.isnan(value) or math.isinf(value):
                    return None
                return float(value)
            # integers
            if isinstance(value, (np.integer, int)):
                return int(value)
            # numpy bool
            if isinstance(value, (np.bool_, bool)):
                return bool(value)
            # numpy arrays
            if isinstance(value, (np.ndarray,)):
                return value.tolist()
            # pandas Timestamp -> isoformat
            try:
                import pandas as _pd
                if isinstance(value, _pd.Timestamp):
                    return value.isoformat()
            except Exception:
                pass
            return value

        # Limpar valores NaN/Infinity de cada alerta
        cleaned_alertas = []
        for alerta in alertas_list:
            cleaned_alerta = {}
            for key, value in alerta.items():
                cleaned_alerta[key] = clean_value(value)
            cleaned_alertas.append(cleaned_alerta)

        return jsonify({
            'alertas': cleaned_alertas,
            'total': len(df_alertas),
            'page': page,
            'totalPages': (len(df_alertas) + size - 1) // size if size > 0 else 0
        })

    except Exception as e:
        print(f"❌ Erro ao listar alertas: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Erro ao carregar alertas: {str(e)}',
            'alertas': [],
            'total': 0,
            'page': 0,
            'totalPages': 0
        }), 500

@app.route('/api/admin/consulta/<int:consulta_id>', methods=['GET'])
@require_auth(role='admin')
def obter_consulta_detalhada(consulta_id):
    """Obtém detalhes completos de uma consulta"""
    try:
        # Debug: verificar se o ID existe
        print(f"🔍 Buscando consulta ID: {consulta_id}")
        
        # Verificar se a coluna 'id' existe e é do tipo correto
        if 'id' not in ml_system.df.columns:
            print("❌ Coluna 'id' não encontrada no DataFrame")
            return jsonify({'error': 'Estrutura de dados inválida'}), 500
        
        # Converter ID para o mesmo tipo da coluna
        ml_system.df['id'] = pd.to_numeric(ml_system.df['id'], errors='coerce')
        
        # Buscar o registro
        registro = ml_system.df[ml_system.df['id'] == consulta_id]
        
        if registro.empty:
            print(f"❌ Consulta {consulta_id} não encontrada")
            return jsonify({'error': 'Consulta não encontrada'}), 404
        
        # Converter para dicionário
        registro_dict = registro.iloc[0].to_dict()
        
        # Tratar valores NaN/None
        for key, value in registro_dict.items():
            if pd.isna(value):
                registro_dict[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                registro_dict[key] = float(value) if isinstance(value, np.floating) else int(value)
        
        # Garantir que CPF seja string
        if 'cpf' in registro_dict and registro_dict['cpf'] is not None:
            registro_dict['cpf'] = str(registro_dict['cpf'])
        
        # Buscar histórico do paciente
        cpf = registro_dict.get('cpf')
        historico = []
        if cpf:
            historico_df = ml_system.df[ml_system.df['cpf'].astype(str) == str(cpf)].sort_values('created_at', ascending=False)
            historico = historico_df.head(10).to_dict('records')
            
            # Limpar valores NaN do histórico
            for h in historico:
                for key, value in h.items():
                    if pd.isna(value):
                        h[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        h[key] = float(value) if isinstance(value, np.floating) else int(value)
        
        # Buscar histórico do dentista
        dentista = registro_dict.get('dentista')
        stats_dentista = {
            'total_procedimentos': 0,
            'suspeitos': 0,
            'taxa_suspeita': 0
        }
        
        if dentista:
            historico_dentista = ml_system.df[ml_system.df['dentista'] == dentista]
            if len(historico_dentista) > 0:
                stats_dentista = {
                    'total_procedimentos': len(historico_dentista),
                    'suspeitos': len(historico_dentista[historico_dentista['status'].isin(['atencao', 'em_analise', 'reprovado', 'pendente', 'fraude'])]),
                    'taxa_suspeita': (len(historico_dentista[historico_dentista['status'].isin(['atencao', 'em_analise', 'reprovado', 'pendente', 'fraude'])]) / len(historico_dentista) * 100) if len(historico_dentista) > 0 else 0
                }
        
        print(f"✅ Consulta {consulta_id} encontrada e processada")
        
        return jsonify({
            'consulta': registro_dict,
            'historico_paciente': historico,
            'stats_dentista': stats_dentista
        })
        
    except Exception as e:
        print(f"❌ Erro ao obter consulta {consulta_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500


@app.route('/api/debug/check-data', methods=['GET'])
@require_auth(role='admin')
def debug_check_data():
    """Endpoint de debug para verificar estrutura dos dados"""
    try:
        info = {
            'total_registros': len(ml_system.df),
            'colunas': list(ml_system.df.columns),
            'tipos_dados': {col: str(dtype) for col, dtype in ml_system.df.dtypes.items()},
            'primeiros_ids': list(ml_system.df['id'].head(10)) if 'id' in ml_system.df.columns else [],
            'status_unicos': list(ml_system.df['status'].unique()) if 'status' in ml_system.df.columns else []
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/classificar/<int:consulta_id>', methods=['POST'])
@require_auth(role='admin')
def classificar_consulta(consulta_id):
    """Admin classifica uma consulta como fraude ou legítima"""
    data = request.json
    classificacao = data.get('classificacao')  # 'fraude' ou 'legitimo'
    observacoes = data.get('observacoes', '')
    
    if classificacao not in ['fraude', 'legitimo']:
        return jsonify({'error': 'Classificação inválida. Use "fraude" ou "legitimo"'}), 400
    
    # Atualizar registro
    idx = ml_system.df[ml_system.df['id'] == consulta_id].index
    if len(idx) == 0:
        return jsonify({'error': 'Consulta não encontrada'}), 404
    
    idx = idx[0]
    
    # Atualizar status
    if classificacao == 'fraude':
        ml_system.df.loc[idx, 'status'] = 'fraude'  # ou 'reprovado'
        ml_system.df.loc[idx, 'fraude'] = 1
    else:
        ml_system.df.loc[idx, 'status'] = 'legitimo'  # ou 'aprovado'
        ml_system.df.loc[idx, 'fraude'] = 0
    
    if observacoes:
        current_motivo = ml_system.df.loc[idx, 'motivo_atencao']
        ml_system.df.loc[idx, 'motivo_atencao'] = f"{current_motivo} | Admin: {observacoes}" if current_motivo else f"Admin: {observacoes}"
    
    # Salvar
    ml_system.df.to_csv(CSV_FILE, index=False)
    
    # ML aprende com a classificação
    modelo_retreinado = ml_system.learn_from_classification(consulta_id, classificacao)
    
    return jsonify({
        'success': True,
        'message': 'Classificação registrada com sucesso',
        'modelo_retreinado': modelo_retreinado,
        'nova_acuracia': ml_system.metrics.get('accuracy', 0)
    })

@app.route('/api/admin/contatar-dentista', methods=['POST'])
@require_auth(role='admin')
def contatar_dentista():
    """
    Endpoint para enviar email de contato ao dentista
    Body: {
        "consulta_id": int,
        "dentista_email": string,
        "mensagem": string
    }
    """
    try:
        data = request.json
        consulta_id = data.get('consulta_id')
        dentista_email = data.get('dentista_email')
        mensagem = data.get('mensagem', '').strip()
        
        # Validações
        if not consulta_id:
            return jsonify({'error': 'ID da consulta é obrigatório'}), 400
        
        if not dentista_email:
            return jsonify({'error': 'Email do dentista é obrigatório'}), 400
        
        if not mensagem:
            return jsonify({'error': 'Mensagem é obrigatória'}), 400
        
        # Buscar consulta
        consulta = ml_system.df[ml_system.df['id'] == consulta_id]
        if consulta.empty:
            return jsonify({'error': 'Consulta não encontrada'}), 404
        
        consulta_dict = consulta.iloc[0].to_dict()
        
        # Montar email
        assunto = f'OdontoVision - Contato sobre Procedimento #{consulta_id}'
        
        corpo_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 20px; border-radius: 10px 10px 0 0; 
                }}
                .content {{ background: #f9f9f9; padding: 20px; border: 1px solid #ddd; }}
                .info-box {{ 
                    background: white; padding: 15px; margin: 10px 0;
                    border-left: 4px solid #667eea; border-radius: 5px;
                }}
                .footer {{ text-align: center; color: #777; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>🦷 OdontoVision</h2>
                    <p>Sistema de Detecção de Fraudes</p>
                </div>
                
                <div class="content">
                    <h3>Olá, {consulta_dict.get('dentista', 'Dentista')},</h3>
                    
                    <p>A administração do OdontoVision entrou em contato sobre um procedimento registrado:</p>
                    
                    <div class="info-box">
                        <h4>📋 Informações do Procedimento:</h4>
                        <p><strong>ID da Consulta:</strong> #{consulta_id}</p>
                        <p><strong>Paciente:</strong> {consulta_dict.get('paciente_nome', 'N/A')}</p>
                        <p><strong>Procedimento:</strong> {consulta_dict.get('procedimento', 'N/A')}</p>
                        <p><strong>Data:</strong> {pd.to_datetime(consulta_dict.get('created_at')).strftime('%d/%m/%Y às %H:%M')}</p>
                        <p><strong>Dente:</strong> {consulta_dict.get('dente', 'N/A')}</p>
                    </div>
                    
                    <div class="info-box">
                        <h4>💬 Mensagem da Administração:</h4>
                        <p>{mensagem}</p>
                    </div>
                    
                    <p>Por favor, entre em contato conosco caso tenha alguma dúvida ou precise esclarecer informações sobre este procedimento.</p>
                    
                    <p><strong>Administrador responsável:</strong> {request.user.get('name', 'Administração')}</p>
                </div>
                
                <div class="footer">
                    <p>Este é um email automático do sistema OdontoVision.</p>
                    <p>Por favor, não responda diretamente a este email.</p>
                    <p>&copy; 2025 OdontoVision - Todos os direitos reservados</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Enviar email
        resultado = enviar_email(dentista_email, assunto, corpo_html)
        
        if not resultado['sucesso']:
            return jsonify({
                'error': 'Erro ao enviar email',
                'message': resultado.get('erro', 'Erro desconhecido')
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Email enviado com sucesso'
        }), 200
    
    except Exception as e:
        print(f'Erro ao contatar dentista: {str(e)}')
        return jsonify({
            'error': 'Erro interno do servidor',
            'message': str(e)
        }), 500

@app.route('/api/admin/relatorio/export', methods=['GET'])
@require_auth(role='admin')
def exportar_relatorio():
    """Exporta relatório em CSV"""
    df = ml_system.df
    
    # Filtros
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    status = request.args.get('status')
    
    if start_date:
        df = df[pd.to_datetime(df['created_at']) >= pd.to_datetime(start_date)]
    if end_date:
        df = df[pd.to_datetime(df['created_at']) <= pd.to_datetime(end_date)]
    if status:
        status_list = [s.strip() for s in status.split(',') if s.strip()]
        if status_list:
            df = df[df['status'].isin(status_list)]
    
    # Gerar CSV
    csv_data = df.to_csv(index=False)
    
    return csv_data, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': f'attachment; filename=relatorio_odontovision_{datetime.now().strftime("%Y%m%d")}.csv'
    }

# ==================== ROTAS PÚBLICAS ====================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'total_registros': len(ml_system.df),
        'modelo_treinado': ml_system.model is not None,
        'acuracia_modelo': ml_system.metrics.get('accuracy', 0)
    })

# ==================== SERVIDOR ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 OdontoVision Backend iniciado na porta {port}")
    print(f"📊 Total de registros: {len(ml_system.df)}")
    print(f"🤖 Modelo ML: {'Ativo' if ml_system.model else 'Inativo'}")
    print(f"🎯 Acurácia atual: {ml_system.metrics.get('accuracy', 0):.2%}")
    print(f"📧 Email configurado: {'Sim' if SMTP_PASSWORD else 'Modo simulação'}\n")
    
    app.run(host='0.0.0.0', port=port, debug=True)