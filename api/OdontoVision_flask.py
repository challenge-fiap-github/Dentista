from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime, timedelta   # << adiciona timedelta
from typing import List, Dict, Any

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

# Coluna status (default = normal para o monitor de fraudes)
if "status" not in raw_df.columns:
    raw_df["status"] = "normal"

# ======================
# Regra: limpeza -> canal em 15 dias  => suspeito
# ======================
def marcar_suspeitas_limpeza_para_canal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "created_at" not in df.columns:
        return df

    # garante datetime
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # coluna booleana
    if "suspicious" not in df.columns:
        df["suspicious"] = False
    else:
        df["suspicious"] = df["suspicious"].fillna(False)

    # para cada CPF, ordena por data e busca padrão limpeza -> canal em 15 dias
    for cpf, g in df.groupby("cpf"):
        g = g.sort_values("created_at")
        for i in range(len(g) - 1):
            p1 = str(g.iloc[i]["procedimento"]).lower()
            p2 = str(g.iloc[i + 1]["procedimento"]).lower()
            d1 = g.iloc[i]["created_at"]
            d2 = g.iloc[i + 1]["created_at"]
            if "profilaxia" in p1 and ("canal" in p2 or "tratamento de canal" in p2):
                if pd.notna(d1) and pd.notna(d2) and (d2 - d1) <= timedelta(days=15):
                    df.loc[g.index[i + 1], "suspicious"] = True
                    # se vier vazio/normal, sobe para atenção
                    if "status" in df.columns:
                        st = str(df.loc[g.index[i + 1], "status"]).strip().lower()
                        if st == "" or st == "normal":
                            df.loc[g.index[i + 1], "status"] = "atencao"

    # caso não exista coluna status, cria a partir de suspicious
    if "status" not in df.columns:
        df["status"] = df["suspicious"].map(lambda x: "atencao" if x else "normal")
    else:
        df["status"] = df["status"].fillna("").replace("", "normal")
        df.loc[df["suspicious"] & (df["status"].str.strip().str.lower() == "normal"), "status"] = "atencao"

    return df

# aplica a regra nos dados carregados
raw_df = marcar_suspeitas_limpeza_para_canal(raw_df)

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
            "cpf": str(row.get("cpf", "—"))
        },
        "dentista": {"nome": str(row.get("dentista", "—"))},
        "procedimento": str(row.get("procedimento", "—")),
        "executado": str(row.get("executado", "")),
        "prescricao": str(row.get("prescricao", "")),
        "status": str(row.get("status", "normal")).lower(),
        "suspicious": bool(row.get("suspicious", False)),   # << expõe flag
        "createdAt": to_iso(row.get("created_at"))
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

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)