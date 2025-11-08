"""
Gerador de dados realistas para sistema OdontoVision
Gera dados de 01/01/2022 até 08/11/2025
Com intervalos realistas entre procedimentos baseados em padrões odontológicos
VERSÃO ATUALIZADA com emails dos dentistas
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuração de seed para reprodutibilidade
np.random.seed(42)
random.seed(42)

# ==================== DADOS REALISTAS ====================

# Nomes brasileiros comuns
NOMES = [
    "Maria Silva", "João Santos", "Ana Costa", "Pedro Oliveira", "Juliana Souza",
    "Carlos Rodrigues", "Fernanda Lima", "Lucas Almeida", "Beatriz Pereira", "Rafael Martins",
    "Camila Ferreira", "Bruno Araújo", "Patricia Ribeiro", "Thiago Barbosa", "Amanda Carvalho",
    "Felipe Gomes", "Larissa Dias", "Rodrigo Nascimento", "Gabriela Castro", "Marcelo Rocha",
    "Mariana Cardoso", "Daniel Monteiro", "Aline Correia", "Ricardo Teixeira", "Vanessa Mendes",
    "Anderson Moreira", "Leticia Vieira", "Gustavo Pinto", "Renata Ramos", "Leonardo Freitas",
    "Carolina Lopes", "Vinicius Fernandes", "Tatiana Duarte", "Henrique Moura", "Priscila Campos",
    "Eduardo Nogueira", "Cristina Azevedo", "Fabio Cunha", "Simone Barros", "Alexandre Pires",
    "Luciana Cavalcanti", "Mauricio Santana", "Adriana Torres", "Diego Batista", "Raquel Melo",
    "Sergio Caldeira", "Monica Neves", "Paulo Soares", "Daniela Farias", "Roberto Castro"
]

# Dentistas com CRO e EMAIL
DENTISTAS = [
    {
        "nome": "Dr. Marcos Antonio Silveira",
        "cro": "CRO-SP 45821",
        "email": "marcos.silveira@odontovision.com.br"
    },
    {
        "nome": "Dra. Patricia Mendes Costa",
        "cro": "CRO-RJ 32156",
        "email": "patricia.costa@odontovision.com.br"
    },
    {
        "nome": "Dr. Roberto Carlos Lima",
        "cro": "CRO-MG 28934",
        "email": "roberto.lima@odontovision.com.br"
    },
    {
        "nome": "Dra. Fernanda Cristina Alves",
        "cro": "CRO-SP 51234",
        "email": "fernanda.alves@odontovision.com.br"
    },
    {
        "nome": "Dr. Eduardo Henrique Santos",
        "cro": "CRO-PR 19876",
        "email": "eduardo.santos@odontovision.com.br"
    },
    {
        "nome": "Dra. Juliana Rodrigues Martins",
        "cro": "CRO-RS 41567",
        "email": "juliana.martins@odontovision.com.br"
    },
    {
        "nome": "Dr. Lucas Gabriel Ferreira",
        "cro": "CRO-BA 36789",
        "email": "lucas.ferreira@odontovision.com.br"
    },
    {
        "nome": "Dra. Camila Beatriz Souza",
        "cro": "CRO-SP 48932",
        "email": "camila.souza@odontovision.com.br"
    },
    {
        "nome": "Dr. Rafael Augusto Pereira",
        "cro": "CRO-RJ 29845",
        "email": "rafael.pereira@odontovision.com.br"
    },
    {
        "nome": "Dra. Mariana Vitoria Carvalho",
        "cro": "CRO-CE 34521",
        "email": "mariana.carvalho@odontovision.com.br"
    }
]

# Procedimentos odontológicos com códigos TUSS
PROCEDIMENTOS = {
    "profilaxia": {
        "nome": "Profilaxia Dentária (Limpeza)",
        "codigo_tuss": "81000030",
        "intervalo_normal_dias": (90, 180),  # 3 a 6 meses
        "intervalo_doenca_dias": (60, 90),   # Com doença periodontal
        "custo_medio": (80, 150)
    },
    "restauracao": {
        "nome": "Restauração em Resina Composta",
        "codigo_tuss": "82000108",
        "intervalo_minimo_dias": 0,  # Pode ser mesmo dia
        "custo_medio": (150, 350)
    },
    "canal": {
        "nome": "Tratamento de Canal (Endodontia)",
        "codigo_tuss": "83000022",
        "sessoes": (1, 3),  # 1 a 3 sessões
        "intervalo_sessoes_dias": (7, 14),  # 1 a 2 semanas entre sessões
        "intervalo_repeticao_dias": 90,  # Mínimo 90 dias para repetir no mesmo dente
        "custo_medio": (600, 1500)
    },
    "extracao": {
        "nome": "Extração Dentária Simples",
        "codigo_tuss": "84000018",
        "intervalo_pos_canal_dias": 90,  # Mínimo 90 dias após canal (senão suspeito)
        "custo_medio": (200, 450)
    },
    "clareamento": {
        "nome": "Clareamento Dentário",
        "codigo_tuss": "82000400",
        "intervalo_normal_dias": (180, 365),  # 6 meses a 1 ano
        "custo_medio": (800, 2000)
    },
    "coroa": {
        "nome": "Coroa Protética",
        "codigo_tuss": "85000045",
        "intervalo_pos_canal_dias": (7, 30),  # 1 semana a 1 mês após canal
        "custo_medio": (1200, 3000)
    },
    "implante": {
        "nome": "Implante Dentário Unitário",
        "codigo_tuss": "86000022",
        "intervalo_pos_extracao_dias": (90, 180),  # 3 a 6 meses após extração
        "custo_medio": (2500, 5000)
    },
    "raspagem": {
        "nome": "Raspagem Periodontal",
        "codigo_tuss": "81000056",
        "intervalo_normal_dias": (90, 180),
        "custo_medio": (300, 600)
    }
}

# Medicamentos prescritos
MEDICAMENTOS = {
    "antibioticos": ["Amoxicilina 500mg", "Amoxicilina + Clavulanato 875mg", "Azitromicina 500mg", 
                     "Cefalexina 500mg", "Metronidazol 400mg"],
    "analgesicos": ["Dipirona 500mg", "Paracetamol 750mg", "Paracetamol 500mg"],
    "antiinflamatorios": ["Ibuprofeno 600mg", "Nimesulida 100mg", "Diclofenaco Potássico 50mg",
                          "Cetoprofeno 100mg"],
    "corticoides": ["Dexametasona 4mg", "Prednisolona 20mg"]
}

# Dentes (numeração FDI)
DENTES = list(range(11, 19)) + list(range(21, 29)) + list(range(31, 39)) + list(range(41, 49))

def gerar_cpf():
    """Gera CPF válido"""
    def calcular_digito(digs):
        s = 0
        for i, d in enumerate(digs):
            s += d * (len(digs) + 1 - i)
        resto = s % 11
        return 0 if resto < 2 else 11 - resto
    
    cpf = [random.randint(0, 9) for _ in range(9)]
    cpf.append(calcular_digito(cpf))
    cpf.append(calcular_digito(cpf))
    return ''.join(map(str, cpf))

def gerar_prescricao(procedimento):
    """Gera prescrição realista baseada no procedimento"""
    proc_tipo = procedimento.lower()
    prescricoes = []
    
    # Procedimentos que geralmente precisam de medicação
    if any(x in proc_tipo for x in ['canal', 'extração', 'extracao', 'cirurg', 'implante']):
        # Antibiótico (70% dos casos)
        if random.random() < 0.7:
            antibiotico = random.choice(MEDICAMENTOS['antibioticos'])
            prescricoes.append(f"{antibiotico} - 1 comp. 8/8h por 7 dias")
        
        # Analgésico (90% dos casos)
        if random.random() < 0.9:
            analgesico = random.choice(MEDICAMENTOS['analgesicos'])
            prescricoes.append(f"{analgesico} - 1 comp. 6/6h se dor")
        
        # Anti-inflamatório (80% dos casos)
        if random.random() < 0.8:
            antiinflam = random.choice(MEDICAMENTOS['antiinflamatorios'])
            prescricoes.append(f"{antiinflam} - 1 comp. 12/12h por 3 dias")
    
    elif 'raspagem' in proc_tipo:
        # Raspagem - às vezes prescreve anti-inflamatório
        if random.random() < 0.5:
            antiinflam = random.choice(MEDICAMENTOS['antiinflamatorios'])
            prescricoes.append(f"{antiinflam} - 1 comp. 12/12h por 3 dias")
    
    # Casos sem prescrição
    if not prescricoes or random.random() < 0.1:
        return "Sem prescrição medicamentosa"
    
    return " | ".join(prescricoes)

def gerar_execucao(procedimento):
    """Gera descrição da execução baseada no procedimento"""
    proc_tipo = procedimento.lower()
    dente = random.choice(DENTES)
    
    execucoes = {
        'profilaxia': f"Realizada profilaxia completa com remoção de tártaro supragengival e polimento",
        'restauracao': f"Restauração em resina composta no dente {dente}, classe II ocluso-mesial",
        'restauração': f"Restauração em resina composta no dente {dente}, classe I oclusal",
        'canal': f"Tratamento endodôntico no dente {dente} - sessão {random.randint(1, 3)}/3",
        'endodontia': f"Tratamento endodôntico no dente {dente} - preparo e obturação dos canais",
        'extração': f"Exodontia do dente {dente} com técnica minimamente invasiva",
        'extracao': f"Exodontia do dente {dente} por indicação de comprometimento radicular",
        'clareamento': f"Clareamento dentário pela técnica de consultório - sessão única",
        'coroa': f"Preparo e moldagem para coroa protética no dente {dente}",
        'implante': f"Instalação de implante dentário na região do dente {dente}",
        'raspagem': f"Raspagem e alisamento radicular nos quadrantes superiores"
    }
    
    for key, exec_text in execucoes.items():
        if key in proc_tipo:
            return exec_text
    
    return f"Procedimento realizado conforme protocolo técnico no dente {dente}"

class HistoricoPaciente:
    """Mantém histórico de procedimentos por paciente"""
    def __init__(self):
        self.historico = {}  # cpf -> lista de procedimentos com datas e dentes
    
    def adicionar(self, cpf, data, procedimento, dente):
        if cpf not in self.historico:
            self.historico[cpf] = []
        self.historico[cpf].append({
            'data': data,
            'procedimento': procedimento,
            'dente': dente
        })
    
    def obter_ultimo_procedimento(self, cpf, tipo_proc):
        """Retorna último procedimento de um tipo específico"""
        if cpf not in self.historico:
            return None
        
        for proc in reversed(self.historico[cpf]):
            if tipo_proc.lower() in proc['procedimento'].lower():
                return proc
        return None
    
    def obter_ultimo_no_dente(self, cpf, dente):
        """Retorna último procedimento em um dente específico"""
        if cpf not in self.historico:
            return None
        
        for proc in reversed(self.historico[cpf]):
            if proc['dente'] == dente:
                return proc
        return None

def identificar_suspeitas(historico_paciente, cpf, data_atual, procedimento_atual, dente_atual):
    """
    Identifica padrões suspeitos baseado em intervalos reais da odontologia
    """
    motivos = []
    
    # Verificar profilaxia muito frequente (< 60 dias)
    if 'profilaxia' in procedimento_atual.lower():
        ultimo_prof = historico_paciente.obter_ultimo_procedimento(cpf, 'profilaxia')
        if ultimo_prof:
            dias = (data_atual - ultimo_prof['data']).days
            if dias < 60:
                motivos.append(f"Profilaxia repetida em {dias} dias (mínimo normal: 90 dias)")
    
    # Verificar canal seguido de extração rápida (< 90 dias mesmo dente)
    if 'extração' in procedimento_atual.lower() or 'extracao' in procedimento_atual.lower():
        ultimo_canal = historico_paciente.obter_ultimo_no_dente(cpf, dente_atual)
        if ultimo_canal and 'canal' in ultimo_canal['procedimento'].lower():
            dias = (data_atual - ultimo_canal['data']).days
            if dias < 90:
                motivos.append(f"Extração {dias} dias após canal no dente {dente_atual} (suspeito < 90 dias)")
    
    # Verificar profilaxia seguida de canal (< 30 dias)
    if 'canal' in procedimento_atual.lower():
        ultimo_prof = historico_paciente.obter_ultimo_procedimento(cpf, 'profilaxia')
        if ultimo_prof:
            dias = (data_atual - ultimo_prof['data']).days
            if dias < 30:
                motivos.append(f"Canal {dias} dias após profilaxia (indica problema pré-existente não detectado)")
    
    # Verificar restauração seguida de canal rápido (< 30 dias mesmo dente)
    if 'canal' in procedimento_atual.lower():
        ultimo_rest = historico_paciente.obter_ultimo_no_dente(cpf, dente_atual)
        if ultimo_rest and 'restaura' in ultimo_rest['procedimento'].lower():
            dias = (data_atual - ultimo_rest['data']).days
            if dias < 30:
                motivos.append(f"Canal {dias} dias após restauração no dente {dente_atual} (suspeito < 30 dias)")
    
    # Verificar clareamento seguido de canal (< 14 dias)
    if 'canal' in procedimento_atual.lower():
        ultimo_clar = historico_paciente.obter_ultimo_procedimento(cpf, 'clareamento')
        if ultimo_clar:
            dias = (data_atual - ultimo_clar['data']).days
            if dias < 14:
                motivos.append(f"Canal {dias} dias após clareamento (altamente suspeito < 14 dias)")
    
    # Verificar múltiplos procedimentos complexos no mesmo dia
    procs_mesmo_dia = [p for p in historico_paciente.historico.get(cpf, []) 
                       if p['data'] == data_atual]
    if len(procs_mesmo_dia) >= 3:
        motivos.append(f"Múltiplos procedimentos no mesmo dia ({len(procs_mesmo_dia)} procedimentos)")
    
    # Procedimentos incompatíveis no mesmo dia
    procs_hoje = [p['procedimento'].lower() for p in procs_mesmo_dia]
    if 'profilaxia' in procs_hoje and any(x in ''.join(procs_hoje) for x in ['extração', 'extracao']):
        motivos.append("Profilaxia e extração no mesmo dia (incompatível)")
    if 'clareamento' in procs_hoje and 'canal' in ''.join(procs_hoje):
        motivos.append("Clareamento e canal no mesmo dia (incompatível)")
    
    return motivos

def gerar_dados_realistas():
    """
    Gera dados realistas de 01/01/2022 a 08/11/2025
    """
    print("🦷 Iniciando geração de dados realistas OdontoVision...")
    print("📅 Período: 01/01/2022 a 08/11/2025")
    
    # Datas
    data_inicio = datetime(2022, 1, 1)
    data_fim = datetime(2025, 11, 8)
    
    # Gerar pacientes únicos
    num_pacientes = 80
    pacientes = []
    for i in range(num_pacientes):
        pacientes.append({
            'nome': random.choice(NOMES),
            'cpf': gerar_cpf(),
            'ultima_consulta': None
        })
    
    dados = []
    historico = HistoricoPaciente()
    
    # Taxa de fraude intencional: 15% dos registros terão padrões suspeitos
    taxa_fraude_intencional = 0.15
    
    # Gerar consultas ao longo do tempo
    data_atual = data_inicio
    id_consulta = 1
    
    while data_atual <= data_fim:
        # Número de consultas por dia (varia entre 2 a 8)
        num_consultas_dia = random.randint(2, 8)
        
        for _ in range(num_consultas_dia):
            # Selecionar paciente (80% pacientes recorrentes, 20% novos)
            if random.random() < 0.8 and len([p for p in pacientes if p['ultima_consulta']]) > 0:
                # Paciente recorrente
                pacientes_com_historico = [p for p in pacientes if p['ultima_consulta']]
                paciente = random.choice(pacientes_com_historico)
            else:
                # Paciente novo ou sem histórico
                paciente = random.choice(pacientes)
            
            # Selecionar dentista
            dentista = random.choice(DENTISTAS)
            
            # Decidir se vai gerar fraude intencional
            gerar_fraude = random.random() < taxa_fraude_intencional
            
            # Selecionar procedimento
            if gerar_fraude:
                # Procedimentos mais propensos a fraude
                tipos_proc = ['profilaxia', 'canal', 'extracao', 'clareamento', 'restauracao']
            else:
                # Distribuição normal de procedimentos
                tipos_proc = list(PROCEDIMENTOS.keys())
            
            tipo_proc = random.choice(tipos_proc)
            procedimento = PROCEDIMENTOS[tipo_proc]['nome']
            codigo_tuss = PROCEDIMENTOS[tipo_proc]['codigo_tuss']
            
            # Gerar dente afetado
            dente = random.choice(DENTES)
            
            # Gerar execução e prescrição
            executado = gerar_execucao(procedimento)
            prescricao = gerar_prescricao(procedimento)
            
            # Se for fraude intencional, forçar intervalo curto
            if gerar_fraude and paciente['ultima_consulta']:
                dias_desde_ultimo = (data_atual - paciente['ultima_consulta']).days
                
                # Forçar procedimentos suspeitos
                if tipo_proc == 'profilaxia' and dias_desde_ultimo < 60:
                    pass  # Já está suspeito
                elif tipo_proc == 'canal':
                    # Forçar canal após procedimentos recentes
                    ultimo = historico.obter_ultimo_procedimento(paciente['cpf'], 'restauracao')
                    if ultimo and (data_atual - ultimo['data']).days < 30:
                        dente = ultimo['dente']  # Mesmo dente
            
            # Identificar suspeitas
            motivos_suspeita = identificar_suspeitas(
                historico, paciente['cpf'], data_atual, procedimento, dente
            )
            
            # Prescrição incompatível (outro tipo de fraude)
            if random.random() < 0.1:  # 10% de prescrições incompatíveis
                if 'profilaxia' in procedimento.lower() or 'clareamento' in procedimento.lower():
                    # Prescrição forte sem necessidade
                    prescricao = f"{random.choice(MEDICAMENTOS['antibioticos'])} - 1 comp. 8/8h por 7 dias"
                    if not motivos_suspeita:
                        motivos_suspeita = []
                    motivos_suspeita.append("Prescrição farmacológica sem procedimento compatível")
            
            # Determinar status e fraude
            is_fraude = 1 if motivos_suspeita else 0
            status = "atencao" if motivos_suspeita else "normal"
            motivo_atencao = " | ".join(motivos_suspeita) if motivos_suspeita else ""
            
            # Adicionar ao dataset
            dados.append({
                'id': id_consulta,
                'paciente_nome': paciente['nome'],
                'cpf': paciente['cpf'],
                'procedimento': procedimento,
                'codigo_tuss': codigo_tuss,
                'executado': executado,
                'prescricao': prescricao,
                'dente': dente,
                'dentista': dentista['nome'],
                'dentista_email': dentista['email'],  # NOVO CAMPO
                'cro_dentista': dentista['cro'],
                'created_at': data_atual.isoformat(),
                'fraude': is_fraude,
                'status': status,
                'motivo_atencao': motivo_atencao,
                'suspicious': is_fraude == 1
            })
            
            # Atualizar histórico
            historico.adicionar(paciente['cpf'], data_atual, procedimento, dente)
            paciente['ultima_consulta'] = data_atual
            
            id_consulta += 1
        
        # Avançar data (pular finais de semana às vezes)
        if data_atual.weekday() == 4:  # Sexta
            data_atual += timedelta(days=3)  # Pular para segunda
        else:
            data_atual += timedelta(days=1)
    
    # Criar DataFrame
    df = pd.DataFrame(dados)
    
    # Estatísticas
    print(f"\n✅ Dados gerados com sucesso!")
    print(f"📊 Total de registros: {len(df)}")
    print(f"🔴 Registros suspeitos: {len(df[df['fraude'] == 1])} ({len(df[df['fraude'] == 1])/len(df)*100:.1f}%)")
    print(f"🟢 Registros normais: {len(df[df['fraude'] == 0])} ({len(df[df['fraude'] == 0])/len(df)*100:.1f}%)")
    print(f"👥 Total de pacientes: {num_pacientes}")
    print(f"👨‍⚕️ Total de dentistas: {len(DENTISTAS)}")
    
    return df

if __name__ == "__main__":
    df = gerar_dados_realistas()
    
    # Salvar CSV usando caminho relativo
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(base_dir, "data", "odonto_vision_data.csv")
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n💾 Arquivo salvo em: {output_file}")
    
    # Mostrar exemplos
    print("\n📋 Exemplos de registros suspeitos:")
    print(df[df['fraude'] == 1][['paciente_nome', 'procedimento', 'created_at', 'motivo_atencao']].head(5))