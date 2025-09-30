from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime
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
            "cpf": "000.000.000-00",
            "procedimento": "Restauração de resina",
            "executado": "Exemplo de execução",
            "prescricao": "Ibuprofeno 400mg",
            "dentista": "Dr. Exemplo",
            "created_at": datetime.now().isoformat(),
            "fraude": 0
        }
    ])

# Garante ID
if "id" not in raw_df.columns:
    raw_df = raw_df.reset_index().rename(columns={"index": "id"})
    raw_df["id"] = raw_df["id"].astype(int) + 1

# Coluna status
if "status" not in raw_df.columns:
    raw_df["status"] = "novo"

# Normaliza data
if "created_at" not in raw_df.columns:
    raw_df["created_at"] = datetime.now().isoformat()

# ======================
# Treinar modelo fraude
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
        "paciente": {"nome": str(row.get("paciente_nome", "—")), "cpf": str(row.get("cpf", "—"))},
        "dentista": {"nome": str(row.get("dentista", "—"))},
        "procedimento": str(row.get("procedimento", "—")),
        "executado": str(row.get("executado", "")),
        "prescricao": str(row.get("prescricao", "")),
        "status": str(row.get("status", "novo")).lower(),
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
    status = (request.args.get("status") or "").lower()
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
    if new_status not in {"novo", "em_analise", "aprovado", "reprovado"}:
        return jsonify({"detail": "status inválido"}), 422

    idx = raw_df.index[raw_df["id"] == diag_id].tolist()
    if not idx:
        return jsonify({"detail": "Diagnóstico não encontrado"}), 404

    raw_df.at[idx[0], "status"] = new_status
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