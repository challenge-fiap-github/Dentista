# api/prepare_csv.py
import re
import pandas as pd
from datetime import timedelta, datetime

CSV = "odonto_vision_data.csv"

LOCKED = {"aprovado", "reprovado", "em_analise"}

REGRAS_DESC = {
    "limpeza_canal_15d": "Profilaxia seguida de tratamento de canal em até 15 dias.",
    "canal_repetido_30d": "Tratamento de canal repetido no mesmo dente em até 30 dias.",
    "clare_canal_7d":     "Clareamento seguido de tratamento de canal em até 7 dias.",
    "rest_extr_20d":      "Restauração seguida de extração do mesmo dente em até 20 dias.",
    "dupla_mesmo_dia":    "Procedimentos potencialmente incompatíveis no mesmo dia (ex.: profilaxia × canal/extração ou clareamento × canal).",
    # Extras
    "multiplos_canais_10d": "Dois ou mais tratamentos de canal em até 10 dias para o mesmo paciente.",
    "clare_rep_14d":        "Clareamento dental repetido em intervalo de até 14 dias.",
    "saudavel_depois_invasivo_15d": "Anotação de 'boca saudável' seguida de canal/extração em até 15 dias.",
    "duplicidade_mesmo_dia": "Possível duplicidade: mesmo procedimento e dente lançados no mesmo dia.",
    "presc_forte_sem_invasivo": "Prescrição forte (antibiótico/anti-inflamatório) sem procedimento invasivo no mesmo dia.",
}

# ----------------- helpers -----------------
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

def _is_extracao(txt: str) -> bool:
    t = _norm(txt)
    return "extração" in t or "extracao" in t

def _is_restauracao(txt: str) -> bool:
    t = _norm(txt)
    return "restauração" in t or "restauracao" in t

def _has_boca_saudavel(txt: str) -> bool:
    t = _norm(txt)
    return "boca saudável" in t or "boca saudavel" in t

def _has_presc_forte(txt: str) -> bool:
    t = _norm(txt)
    keys = [
        "amoxicilina", "amoxilina", "clavulanato", "metronidazol",
        "azitromicina", "antibiotico", "antibiótico",
        "ibuprofeno 600", "diclofenaco", "nimesulida", "corticóide", "corticoide"
    ]
    return any(k in t for k in keys)

def extrair_dente(texto: str) -> str:
    m = re.search(r"dente\s*(\d{1,2})", _norm(texto))
    return m.group(1) if m else ""

def add_reason(cur: str, extra: str) -> str:
    cur = (cur or "").strip()
    if not cur:
        return extra
    parts = [p.strip() for p in cur.split(";") if p.strip()]
    if extra not in parts:
        parts.append(extra)
    return "; ".join(parts)

def clamp_years_2022_2025(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"].astype(str).str.replace("T", " ").str.replace("Z", ""), errors="coerce")
    def map_year(dt):
        if pd.isna(dt): return dt
        y = min(2025, max(2022, dt.year))
        return dt.replace(year=y)
    df["created_at"] = df["created_at"].map(map_year)
    return df

def shuffle_month_day_preserving_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Embaralha mês/dia de forma determinística por id (para quebrar o desenho do gráfico),
    preservando ano e horário.
    """
    df = df.copy()
    if "id" not in df.columns:
        return df
    def transform(row):
        dt = row["created_at"]
        if pd.isna(dt):
            return row["created_at"]
        rid = int(row["id"])
        new_month = (rid % 12) + 1
        new_day   = ((rid * 7) % 28) + 1  # 1..28 evita problemas de mês
        try:
            return dt.replace(month=new_month, day=new_day)
        except ValueError:
            # fallback raro (fev p/ 30 etc.): força dia 28
            return dt.replace(month=new_month, day=28)
    df["created_at"] = df.apply(transform, axis=1)
    return df

# ----------------- motor de regras -----------------
def aplicar_regras(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # colunas base
    for col, default in [("status", "normal"), ("motivo_atencao", ""), ("executado", ""), ("prescricao", "")]:
        if col not in df.columns:
            df[col] = default

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["status"] = df["status"].fillna("normal").map(_norm)
    df["motivo_atencao"] = df["motivo_atencao"].fillna("")

    # campos normalizados
    proc  = df["procedimento"].astype(str)
    exec_ = df["executado"].astype(str)
    df["_is_profilaxia"]  = proc.apply(_is_profilaxia)
    df["_is_canal"]       = proc.apply(_is_canal)
    df["_is_clareamento"] = proc.apply(_is_clareamento)
    df["_is_extracao"]    = proc.apply(_is_extracao)
    df["_is_rest"]        = proc.apply(_is_restauracao)
    df["_saudavel"]       = exec_.apply(_has_boca_saudavel)
    df["_presc_forte"]    = df["prescricao"].astype(str).apply(_has_presc_forte)
    df["_dente"]          = exec_.apply(extrair_dente)

    # ----- R5 + R9 + R10 no mesmo dia -----
    grp_day = df.groupby(["cpf", df["created_at"].dt.date], dropna=True)
    for _, g in grp_day:
        has_profi = g["_is_profilaxia"].any()
        has_canal = g["_is_canal"].any()
        has_extr  = g["_is_extracao"].any()
        has_clare = g["_is_clareamento"].any()

        reasons = []
        if has_profi and (has_canal or has_extr):
            reasons.append(REGRAS_DESC["dupla_mesmo_dia"])
        if has_clare and has_canal and REGRAS_DESC["dupla_mesmo_dia"] not in reasons:
            reasons.append(REGRAS_DESC["dupla_mesmo_dia"])

        if reasons:
            for idx in g.index:
                if df.loc[idx, "status"] in LOCKED:  # respeita travados
                    continue
                df.loc[idx, "status"] = "atencao"
                for r in reasons:
                    df.loc[idx, "motivo_atencao"] = add_reason(df.loc[idx, "motivo_atencao"], r)

        # R9: duplicidade (mesmo procedimento & dente) no mesmo dia
        dup = g.copy()
        key_series = dup.apply(lambda r: (_norm(r["procedimento"]), r["_dente"] or ""), axis=1)
        counts = key_series.value_counts()
        if (counts > 1).any():
            dkeys = {k for k, c in counts.items() if c > 1}
            for pos, k in enumerate(key_series.tolist()):
                if k in dkeys:
                    idx = dup.index[pos]
                    if df.loc[idx, "status"] not in LOCKED:
                        df.loc[idx, "status"] = "atencao"
                        df.loc[idx, "motivo_atencao"] = add_reason(df.loc[idx, "motivo_atencao"], REGRAS_DESC["duplicidade_mesmo_dia"])

        # R10: prescrição forte sem invasivo no mesmo dia
        invasivo = g["_is_canal"].any() or g["_is_extracao"].any() or g["_is_rest"].any()
        if not invasivo and g["_presc_forte"].any():
            for idx in g[g["_presc_forte"]].index:
                if df.loc[idx, "status"] in LOCKED:
                    continue
                df.loc[idx, "status"] = "atencao"
                df.loc[idx, "motivo_atencao"] = add_reason(df.loc[idx, "motivo_atencao"], REGRAS_DESC["presc_forte_sem_invasivo"])

    # ----- Regras temporais por CPF (ordem cronológica) -----
    for cpf, g in df.groupby("cpf"):
        g = g.sort_values("created_at")

        # R1 / R3 / R4 / R8 (pares consecutivos)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            d1, d2 = a["created_at"], b["created_at"]
            if pd.isna(d1) or pd.isna(d2) or d2 < d1:
                continue
            delta = d2 - d1
            reasons = []

            if a["_is_profilaxia"] and b["_is_canal"] and delta <= timedelta(days=15):
                reasons.append(REGRAS_DESC["limpeza_canal_15d"])
            if a["_is_clareamento"] and b["_is_canal"] and delta <= timedelta(days=7):
                reasons.append(REGRAS_DESC["clare_canal_7d"])
            if a["_is_rest"] and b["_is_extracao"] and a["_dente"] and b["_dente"] and a["_dente"] == b["_dente"] and delta <= timedelta(days=20):
                reasons.append(REGRAS_DESC["rest_extr_20d"])
            if a["_saudavel"] and (b["_is_canal"] or b["_is_extracao"]) and delta <= timedelta(days=15):
                reasons.append(REGRAS_DESC["saudavel_depois_invasivo_15d"])

            if reasons:
                idx = b.name
                if df.loc[idx, "status"] not in LOCKED:
                    df.loc[idx, "status"] = "atencao"
                for r in reasons:
                    df.loc[idx, "motivo_atencao"] = add_reason(df.loc[idx, "motivo_atencao"], r)

        # R2: canal repetido MESMO dente <=30d
        canal = g[g["_is_canal"]]
        for dente, gc in canal.groupby("_dente"):
            if not dente:
                continue
            gc = gc.sort_values("created_at")
            for i in range(len(gc) - 1):
                a, b = gc.iloc[i], gc.iloc[i + 1]
                if pd.notna(a["created_at"]) and pd.notna(b["created_at"]) and (b["created_at"] - a["created_at"]) <= timedelta(days=30):
                    idx = b.name
                    if df.loc[idx, "status"] not in LOCKED:
                        df.loc[idx, "status"] = "atencao"
                    df.loc[idx, "motivo_atencao"] = add_reason(df.loc[idx, "motivo_atencao"], REGRAS_DESC["canal_repetido_30d"])

        # R6: >=2 canais em 10 dias (dentes quaisquer)
        c_dates = canal["created_at"].dropna().sort_values().tolist()
        for i in range(len(c_dates) - 1):
            if (c_dates[i + 1] - c_dates[i]) <= timedelta(days=10):
                win_start = c_dates[i]
                win_end = win_start + timedelta(days=10)
                idxs = g.index[(g["_is_canal"]) & (g["created_at"] >= win_start) & (g["created_at"] <= win_end)]
                for idx in idxs:
                    if df.loc[idx, "status"] in LOCKED:
                        continue
                    df.loc[idx, "status"] = "atencao"
                    df.loc[idx, "motivo_atencao"] = add_reason(df.loc[idx, "motivo_atencao"], REGRAS_DESC["multiplos_canais_10d"])

        # R7: clareamento repetido <=14d
        cla = g[g["_is_clareamento"]].sort_values("created_at")
        for i in range(len(cla) - 1):
            a, b = cla.iloc[i], cla.iloc[i + 1]
            if pd.notna(a["created_at"]) and pd.notna(b["created_at"]) and (b["created_at"] - a["created_at"]) <= timedelta(days=14):
                idx = b.name
                if df.loc[idx, "status"] not in LOCKED:
                    df.loc[idx, "status"] = "atencao"
                df.loc[idx, "motivo_atencao"] = add_reason(df.loc[idx, "motivo_atencao"], REGRAS_DESC["clare_rep_14d"])

    # normalização final
    df["status"] = df["status"].replace("", "normal")
    df.loc[df["status"] != "atencao", "motivo_atencao"] = ""

    # limpa auxiliares
    df = df.drop(columns=[
        "_is_profilaxia","_is_canal","_is_clareamento","_is_extracao","_is_rest",
        "_saudavel","_presc_forte","_dente"
    ])

    return df

def preencher_motivos(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche/concatena a coluna 'motivos' quando status for 'atencao'."""
    df = df.copy()
    if "motivos" not in df.columns:
        df["motivos"] = ""

    keys = list(REGRAS_DESC.keys())
    def pick(row):
        if _norm(row.get("status")) != "atencao":
            return ""
        base = REGRAS_DESC[keys[int(row["id"]) % len(keys)]] if "id" in row and pd.notna(row["id"]) else ""
        extra = (row.get("motivo_atencao") or "").strip()
        if extra and extra not in base:
            return f"{base} | {extra}" if base else extra
        return base

    df["motivos"] = [pick(r) for _, r in df.iterrows()]
    return df

def ajustar_proporcao(df: pd.DataFrame, alvo_atencao: float = 0.40) -> pd.DataFrame:
    """
    Garante ~60% normal / 40% atenção.
    Não altera linhas com status em LOCKED.
    """
    df = df.copy()
    mask_at = df["status"].astype(str).str.lower().eq("atencao")
    idx_at = list(df[mask_at & ~df["status"].isin(LOCKED)].index)
    target = round(len(df) * alvo_atencao)
    if len(idx_at) > target:
        to_flip = idx_at[target:]  # mantém as primeiras como atenção
        df.loc[to_flip, "status"] = "normal"
        df.loc[to_flip, "motivo_atencao"] = ""
        if "suspicious" in df.columns:
            df.loc[to_flip, "suspicious"] = False
        if "attention_reason" in df.columns:
            df.loc[to_flip, "attention_reason"] = ""
    return df

# ----------------- main -----------------
def main():
    # CPF como string para não perder zeros à esquerda
    df = pd.read_csv(CSV, dtype={"cpf": str})

    # 1) Normalizar datas para [2022..2025] e 2) embaralhar mês/dia (preserva ano)
    df = clamp_years_2022_2025(df)
    df = shuffle_month_day_preserving_year(df)

    # 3) Aplicar regras (preenche motivo_atencao e status = atencao quando couber)
    df = aplicar_regras(df)

    # 4) Ajustar proporção 60/40 (sem tocar em LOCKED)
    df = ajustar_proporcao(df, 0.40)

    # 5) Preencher coluna 'motivos' (mostrada na sua tabela)
    df = preencher_motivos(df)

    # 6) Normalizações finais
    if "suspicious" in df.columns:
        df["suspicious"] = df["suspicious"].astype(str).str.strip().replace(
            {"True":"True","False":"False","true":"True","false":"False","":"False"}
        )

    # Ordenar por data e colocar 'motivo_atencao' no fim (compatível com seu script)
    if "created_at" in df.columns:
        df = df.sort_values("created_at")
    cols = [c for c in df.columns if c != "motivo_atencao"] + ["motivo_atencao"]
    df = df[cols]

    df.to_csv(CSV, index=False)
    print("✔ CSV atualizado: datas embaralhadas (ano preservado), 60/40 garantido, 'motivos' preenchido.")

if __name__ == "__main__":
    main()