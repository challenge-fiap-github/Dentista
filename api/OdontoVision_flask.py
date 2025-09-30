from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime, timedelta   # << adiciona timedelta
from typing import List, Dict, Any
import re

app = Flask(__name__)
# Libera CORS para rotas /api/*
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ======================
# Carregar dados
# ======================
CSV_FILE = 'odonto_vision_data.csv'
try:
    raw_df = pd.read_csv(CSV_FILE)
except Exception:
    raw_df = pd.DataFrame([
        {
            "id": 1,
            "paciente_nome": "Paciente Exemplo",
            "cpf": "00000000000",
            "procedimento": "Restauração de resina",
            "executado": "Exemplo de execução",
            "prescricao": "Ibuprofeno 400mg",
            "dentista": "Dr. Exemplo",
            "created_at": datetime.now().isoformat(),
            "fraude": 0,
            "status": "normal"
        }
    ])

# Garante ID
if "id" not in raw_df.columns:
    raw_df = raw_df.reset_index().rename(columns={"index": "id"})
    raw_df["id"] = raw_df["id"].astype(int) + 1

# Normaliza data
if "created_at" not in raw_df.columns:
    raw_df["created_at"] = datetime.now().isoformat()

if "status" not in raw_df.columns:
    raw_df["status"] = "normal"

# >>> ADICIONE:
if "motivo_atencao" not in raw_df.columns:
    raw_df["motivo_atencao"] = ""

if "suspicious" not in raw_df.columns:
    raw_df["suspicious"] = False

# ======================
# Treinar modelo fraude (se houver coluna numérica 'fraude')
# ======================
HAS_LABEL = "fraude" in raw_df.columns
FEATURES: List[str] = []
model = None

if HAS_LABEL:
    numeric_df = raw_df.select_dtypes(include=[np.number]).copy()
    if "fraude" in numeric_df.columns:
        X = numeric_df.drop(columns=["fraude"])
        y = numeric_df["fraude"].astype(int)
        FEATURES = list(X.columns)
        if len(FEATURES) > 0 and y.nunique() > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, 'modelo_fraude.pkl')

# ======================
# Helpers
# ======================
def to_iso(dt_val) -> str:
    try:
        return pd.to_datetime(dt_val).isoformat()
    except Exception:
        return datetime.now().isoformat()

def normalize_row(row: pd.Series) -> Dict[str, Any]:
    return {
        "id": int(row.get("id")),
        "paciente": {
            "nome": str(row.get("paciente_nome", "—")),
            "cpf": str(row.get("cpf", "—")),
        },
        "dentista": {"nome": str(row.get("dentista", "—"))},
        "procedimento": str(row.get("procedimento", "—")),
        "executado": str(row.get("executado", "")),
        "prescricao": str(row.get("prescricao", "")),
        "status": str(row.get("status", "normal")).lower(),
        "suspicious": bool(row.get("suspicious", False)),
        "motivoAtencao": str(row.get("motivo_atencao", "")),   # <<< AQUI
        "createdAt": to_iso(row.get("created_at")),
    }

def paginate(items: List[Dict[str, Any]], page: int, size: int):
    total = len(items)
    start = page * size
    end = start + size
    total_pages = max(1, (total + size - 1) // size)
    return {"content": items[start:end], "totalPages": total_pages, "number": page}

# ======================
# Rotas da API
# ======================
@app.get("/api/diagnosticos")
def listar_diagnosticos():
    search = (request.args.get("search") or "").lower()
    status = (request.args.get("status") or "").lower()  # aceita normal/atencao/novo/etc
    page = int(request.args.get("page", 0))
    size = int(request.args.get("size", 10))

    df = raw_df.copy()

    if search:
        mask = (
            df["paciente_nome"].astype(str).str.lower().str.contains(search, na=False)
            | df["cpf"].astype(str).str.lower().str.contains(search, na=False)
            | df["procedimento"].astype(str).str.lower().str.contains(search, na=False)
        )
        df = df[mask]

    if status:
        df = df[df["status"].astype(str).str.lower() == status]

    items = [normalize_row(r) for _, r in df.iterrows()]
    return jsonify(paginate(items, page, size))

@app.get("/api/diagnosticos/<int:diag_id>")
def obter_diagnostico(diag_id: int):
    row = raw_df[raw_df["id"] == diag_id]
    if row.empty:
        return jsonify({"detail": "Diagnóstico não encontrado"}), 404
    return jsonify(normalize_row(row.iloc[0]))

@app.patch("/api/diagnosticos/<int:diag_id>/status")
def alterar_status(diag_id: int):
    body = request.get_json(force=True, silent=True) or {}
    new_status = (body.get("status") or "").lower()

    # >>> aqui está a lista de status aceitos (incluindo normal/atencao)
    if new_status not in {"novo", "em_analise", "aprovado", "reprovado", "normal", "atencao"}:
        return jsonify({"detail": "status inválido"}), 422

    idx = raw_df.index[raw_df["id"] == diag_id].tolist()
    if not idx:
        return jsonify({"detail": "Diagnóstico não encontrado"}), 404

    raw_df.at[idx[0], "status"] = new_status

    # sincroniza a flag de suspeita com o status
    if new_status == "normal":
        raw_df.at[idx[0], "suspicious"] = False
    if new_status == "atencao":
        raw_df.at[idx[0], "suspicious"] = True

    return jsonify(normalize_row(raw_df.iloc[idx[0]]))

@app.post("/api/prever")
def prever_fraude():
    try:
        dados = request.get_json(force=True)
        modelo = joblib.load('modelo_fraude.pkl')
        df_input = pd.DataFrame([dados], columns=FEATURES).fillna(0)
        previsao = int(modelo.predict(df_input)[0])
        return jsonify({
            "fraude_detectada": bool(previsao),
            "mensagem": "Fraude detectada!" if previsao else "Transação segura."
        })
    except Exception as e:
        return jsonify({"erro": str(e)}), 400

@app.get("/api/ping")
def ping():
    return jsonify(ok=True, msg="pong")

@app.post("/api/consultas")
def criar_consulta():
    """
    Recebe o registro do dentista e grava no CSV/DataFrame.
    Já recalcula as regras de suspeita para o CPF.
    """
    global raw_df
    body = request.get_json(force=True, silent=True) or {}

    paciente_nome = str(body.get("paciente_nome", "")).strip()
    cpf = re.sub(r"\D", "", str(body.get("cpf", "")))
    procedimento = str(body.get("procedimento", "")).strip()
    executado = str(body.get("executado", "")).strip()
    prescricao = str(body.get("prescricao", "")).strip()
    dentista = str(body.get("dentista", "Dentista")).strip() or "Dentista"
    created_at = body.get("created_at") or datetime.now().isoformat()
    motivos = body.get("motivos", [])  # array opcional

    # validações simples
    if not paciente_nome or not cpf or not procedimento or not executado:
        return jsonify({"detail": "Campos obrigatórios ausentes."}), 422

    # garante colunas novas
    for col in ["motivos", "motivo_atencao", "suspicious", "status"]:
        if col not in raw_df.columns:
            raw_df[col] = "" if col != "suspicious" else False
    if "fraude" not in raw_df.columns:
        raw_df["fraude"] = 0

    # novo id
    new_id = (int(raw_df["id"].max()) + 1) if ("id" in raw_df.columns and not raw_df.empty) else 1

    # monta linha
    row = {
        "id": new_id,
        "paciente_nome": paciente_nome,
        "cpf": cpf,
        "procedimento": procedimento,
        "executado": executado,
        "prescricao": prescricao,
        "dentista": dentista,
        "created_at": created_at,
        "fraude": 0,
        "status": "normal",
        "motivos": "; ".join(motivos) if isinstance(motivos, list) else str(motivos),
    }

    # anexa
    raw_df = pd.concat([raw_df, pd.DataFrame([row])], ignore_index=True)

    # reavalia suspeita (todas as regras para o CPF)
    try:
        raw_df = _reprocess_rules_for_cpf(raw_df, cpf)
    except Exception as e:
        print("Warn marcar suspeita:", e)

    # persiste
    try:
        raw_df.to_csv(CSV_FILE, index=False)
    except Exception as e:
        print("Warn ao salvar CSV:", e)

    # retorna a linha normalizada
    return jsonify(normalize_row(raw_df.iloc[-1])), 201

# ======================
# Regras de Negócio
# ======================
TOOTH_RE = re.compile(r'dente\s*(\d{1,2})', flags=re.IGNORECASE)

ANTIBIOTICOS = ['amoxicilina', 'clavulanato', 'azitro', 'azitromicina',
                'cefalexina', 'doxiciclina', 'metronidazol']
ANALGESICOS  = ['dipirona', 'paracetamol']
ANTIINFLAM   = ['ibuprofeno', 'diclofenaco', 'naproxeno', 'corticostero', 'prednis']

MEDS_KEYWORDS = ANTIBIOTICOS + ANALGESICOS + ANTIINFLAM

def _kw(s):  # string segura
    return str(s or '').strip().lower()

def _to_dt(s):
    return pd.to_datetime(s, errors='coerce')

def _tooth(text):
    m = TOOTH_RE.search(_kw(text))
    return m.group(1) if m else None

def marcar_suspeitas_multiplas_regras(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "created_at" not in df.columns:
        return df

    # normalizações
    df["created_at"] = _to_dt(df["created_at"])
    for col in ["procedimento", "executado", "prescricao", "dentista", "paciente_nome"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    # colunas alvo
    if "suspicious" not in df.columns:
        df["suspicious"] = False
    else:
        df["suspicious"] = df["suspicious"].fillna(False)

    # Troque attention_reason por motivo_atencao
    if "motivo_atencao" not in df.columns:
        df["motivo_atencao"] = ""
    else:
        df["motivo_atencao"] = df["motivo_atencao"].fillna("")

    # processa por paciente (cpf)
    for cpf, g in df.groupby("cpf"):
        g = g.sort_values("created_at")
        idxs = list(g.index)

        for i in range(len(g)):
            reasons = []

            # dados atuais
            pi = _kw(g.iloc[i]["procedimento"])
            ei = _kw(g.iloc[i]["executado"])
            di = g.iloc[i]["created_at"]
            ti = _tooth(ei) or _tooth(pi)

            # ---------- Regra 6: prescrição forte sem procedimento compatível ----------
            presc = _kw(g.iloc[i]["prescricao"])
            if any(k in presc for k in MEDS_KEYWORDS):
                compat = any(k in pi for k in ["canal", "extraç", "raspagem", "cirurg"])
                if not compat:
                    reasons.append("Prescrição farmacológica sem procedimento clínico compatível")

            # relacionais com registros vizinhos
            if i < len(g) - 1:
                pj = _kw(g.iloc[i+1]["procedimento"])
                ej = _kw(g.iloc[i+1]["executado"])
                dj = g.iloc[i+1]["created_at"]
                tj = _tooth(ej) or _tooth(pj)

                if pd.notna(di) and pd.notna(dj):
                    delta = dj - di

                    # ---------- Regra 1: profilaxia -> canal ≤15d ----------
                    if "profilaxia" in pi and ("canal" in pj or "tratamento de canal" in pj):
                        if delta <= timedelta(days=15):
                            reasons.append(f"Profilaxia em {di.date()} → Canal em {dj.date()} (≤ 15 dias)")

                    # ---------- Regra 2: restauração/resina -> canal mesmo dente ≤20d ----------
                    if ("restaura" in pi or "resina" in pi) and ("canal" in pj):
                        if delta <= timedelta(days=20) and (ti and tj and ti == tj):
                            reasons.append(f"Restauração dente {ti} → Canal {tj} (≤ 20 dias)")

                    # ---------- Regra 3: canal -> extração mesmo dente ≤30d ----------
                    if "canal" in pi and ("extraç" in pj):
                        if delta <= timedelta(days=30) and (ti and tj and ti == tj):
                            reasons.append(f"Canal dente {ti} → Extração {tj} (≤ 30 dias)")

                    # ---------- Regra 5: troca de dentista em ≤7d p/ procedimento semelhante ----------
                    dent_i = _kw(g.iloc[i]["dentista"])
                    dent_j = _kw(g.iloc[i+1]["dentista"])
                    if dent_i and dent_j and dent_i != dent_j and delta <= timedelta(days=7):
                        sem = any(k in pi and k in pj for k in ["canal", "restaura", "resina", "profilaxia", "extraç", "coroa", "prótese", "raspagem"])
                        if sem:
                            reasons.append(f"Troca de dentista em {delta.days} dia(s) para procedimento semelhante")

                    # ---------- Regra 7: dois canais diferentes em ≤30d ----------
                    if "canal" in pi and "canal" in pj and (ti != tj) and delta <= timedelta(days=30):
                        reasons.append(f"Dois canais próximos (dentes {ti or '?'} e {tj or '?'}) em ≤ 30 dias")

            # ---------- Regra 4: 3+ atendimentos em ≤14d ----------
            cnt14 = 1
            for k in range(i+1, len(g)):
                if pd.notna(g.iloc[k]["created_at"]) and pd.notna(di):
                    if (g.iloc[k]["created_at"] - di) <= timedelta(days=14):
                        cnt14 += 1
            if cnt14 >= 3:
                reasons.append(f"{cnt14} atendimentos em ≤ 14 dias")

            # aplica no dataframe
            if reasons:
                j = idxs[i]
                df.loc[j, "suspicious"] = True
                # agrega com o que já tinha
                prev = _kw(df.loc[j, "motivo_atencao"])
                all_reasons = " | ".join([r for r in [prev] + reasons if r])
                df.loc[j, "motivo_atencao"] = all_reasons
                st = _kw(df.loc[j, "status"]) if "status" in df.columns else ""
                if st in ("", "normal", "novo"):
                    df.loc[j, "status"] = "atencao"

    # fallback de status
    if "status" not in df.columns:
        df["status"] = df["suspicious"].map(lambda x: "atencao" if x else "normal")
    else:
        df["status"] = df["status"].fillna("").replace("", "normal")
        df.loc[df["suspicious"] & (df["status"].str.lower() == "normal"), "status"] = "atencao"

    return df

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _is_profilaxia(txt: str) -> bool:
    t = _norm(txt)
    return "profilaxia" in t or "limpeza" in t

def _is_canal(txt: str) -> bool:
    t = _norm(txt)
    return "tratamento de canal" in t or "canal" in t

def _is_clareamento(txt: str) -> bool:
    return "clareamento" in _norm(txt)

def _is_restauracao(txt: str) -> bool:
    return "restauração" in _norm(txt) or "restauracao" in _norm(txt)

def _is_extracao(txt: str) -> bool:
    t = _norm(txt)
    return "extração" in t or "extracao" in t

def _append_reason(cur: str, extra: str) -> str:
    cur = (cur or "").strip()
    if not cur:
        return extra
    # evita duplicados simples
    parts = [p.strip() for p in cur.split(";") if p.strip()]
    if extra not in parts:
        parts.append(extra)
    return "; ".join(parts)

def _reprocess_rules_for_cpf(df: pd.DataFrame, cpf: str) -> pd.DataFrame:
    """
    Reprocessa SOMENTE os registros do CPF informado e aplica as regras:
      R1: Profilaxia -> Canal em <=15 dias
      R2: Canal repetido em <=30 dias
      R3: Clareamento -> Canal em <=7 dias
      R4: Restauração -> Extração em <=20 dias
      R5: Conflitos no mesmo dia (Profilaxia + Extração, Clareamento + Canal)
    """
    from datetime import timedelta

    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    if "suspicious" not in df.columns:
        df["suspicious"] = False
    if "motivo_atencao" not in df.columns:
        df["motivo_atencao"] = ""

    locked = {"aprovado", "reprovado", "em_analise"}

    # pega somente os índices daquele CPF
    mask = df["cpf"].astype(str) == str(cpf)
    idxs = df.index[mask]
    if not len(idxs):
        return df

    # Zera flags desse recorte, mas mantém status travados
    df.loc[idxs, "suspicious"] = False
    # NÃO limpamos motivo_atencao aqui porque podemos compor; vamos reconstruí-lo
    df.loc[idxs, "motivo_atencao"] = ""

    # Ordena por data no recorte
    g = df.loc[idxs].sort_values("created_at")

    # ---------- R5: Conflitos no mesmo dia ----------
    # agrupar por dia
    g["dia"] = g["created_at"].dt.date
    for dia, dayblock in g.groupby("dia"):
        procs = dayblock["procedimento"].astype(str).tolist()
        has_profi = any(_is_profilaxia(p) for p in procs)
        has_extr  = any(_is_extracao(p) for p in procs)
        has_cla   = any(_is_clareamento(p) for p in procs)
        has_canal = any(_is_canal(p) for p in procs)

        # conflitantes
        reasons = []
        if has_profi and has_extr:
            reasons.append("Conflito no mesmo dia: profilaxia e extração.")
        if has_cla and has_canal:
            reasons.append("Conflito no mesmo dia: clareamento e tratamento de canal.")

        if reasons:
            for j in dayblock.index:
                df.loc[j, "suspicious"] = True
                df.loc[j, "motivo_atencao"] = _append_reason(df.loc[j, "motivo_atencao"], "; ".join(reasons))
                st = _norm(df.loc[j, "status"])
                if st not in locked:
                    df.loc[j, "status"] = "atencao"

    # ---------- Regras de janelas temporais (pares ordenados) ----------
    glist = g.reset_index()  # tem col 'index' com o índice original
    for i in range(len(glist) - 1):
        idx_i = glist.loc[i, "index"]
        idx_j = glist.loc[i + 1, "index"]

        p1 = str(df.loc[idx_i, "procedimento"])
        p2 = str(df.loc[idx_j, "procedimento"])
        d1 = df.loc[idx_i, "created_at"]
        d2 = df.loc[idx_j, "created_at"]

        if pd.isna(d1) or pd.isna(d2) or d2 < d1:
            continue

        delta = d2 - d1

        reasons = []

        # R1: profilaxia -> canal <=15d
        if _is_profilaxia(p1) and _is_canal(p2) and delta <= timedelta(days=15):
            reasons.append("Profilaxia seguida de tratamento de canal em até 15 dias.")

        # R2: canal repetido <=30d
        if _is_canal(p1) and _is_canal(p2) and delta <= timedelta(days=30):
            reasons.append("Tratamento de canal repetido em até 30 dias.")

        # R3: clareamento -> canal <=7d
        if _is_clareamento(p1) and _is_canal(p2) and delta <= timedelta(days=7):
            reasons.append("Clareamento seguido de tratamento de canal em até 7 dias.")

        # R4: restauração -> extração <=20d
        if _is_restauracao(p1) and _is_extracao(p2) and delta <= timedelta(days=20):
            reasons.append("Restauração seguida de extração em até 20 dias.")

        if reasons:
            # marcamos no segundo atendimento (idx_j)
            df.loc[idx_j, "suspicious"] = True
            for r in reasons:
                df.loc[idx_j, "motivo_atencao"] = _append_reason(df.loc[idx_j, "motivo_atencao"], r)
            st = _norm(df.loc[idx_j, "status"])
            if st not in locked:
                df.loc[idx_j, "status"] = "atencao"

    # preenche vazios com normal se nada suspeito e não travado
    for j in g.index:
        if not bool(df.loc[j, "suspicious"]):
            st = _norm(df.loc[j, "status"])
            if st not in locked and st != "normal":
                # se ficou algo tipo "", corrige para normal
                df.loc[j, "status"] = "normal"
            # limpa motivo se não está suspeito
            if not df.loc[j, "suspicious"]:
                df.loc[j, "motivo_atencao"] = df.loc[j, "motivo_atencao"] or ""

    return df
# aplica as regras
raw_df = marcar_suspeitas_multiplas_regras(raw_df)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)