# api/prepare_csv.py
import re
import pandas as pd
from datetime import timedelta
from typing import List

CSV = "odonto_vision_data.csv"

# Status que não devem ser alterados automaticamente
LOCKED = {"aprovado", "reprovado", "em_analise"}

REGRAS_DESC = {
    "limpeza_canal_15d":        "Profilaxia seguida de tratamento de canal em até 15 dias.",
    "canal_repetido_30d":       "Tratamento de canal repetido no mesmo dente em até 30 dias.",
    "clare_canal_7d":           "Clareamento seguido de tratamento de canal em até 7 dias.",
    "rest_extr_20d":            "Restauração seguida de extração do mesmo dente em até 20 dias.",
    "dupla_mesmo_dia":          "Procedimentos potencialmente incompatíveis no mesmo dia (ex.: profilaxia × canal/extração ou clareamento × canal).",
    # extras
    "multiplos_canais_10d":     "Dois ou mais tratamentos de canal em até 10 dias para o mesmo paciente.",
    "clare_rep_14d":            "Clareamento dental repetido em intervalo de até 14 dias.",
    "saudavel_depois_invasivo": "Anotação de 'boca saudável' seguida de canal/extração em até 15 dias.",
    "duplicidade_mesmo_dia":    "Possível duplicidade: mesmo procedimento e dente lançados no mesmo dia.",
    "presc_forte_sem_invasivo": "Prescrição forte (antibiótico/anti-inflamatório) sem procedimento invasivo no mesmo dia.",
}

# -------------------------------------------------
# Helpers de normalização
# -------------------------------------------------
def _norm(s: str) -> str:
    return (str(s) if s is not None else "").strip().lower()

def _is_profilaxia(txt: str) -> bool:
    t = _norm(txt); return "profilaxia" in t or "limpeza" in t

def _is_canal(txt: str) -> bool:
    t = _norm(txt); return "tratamento de canal" in t or "canal" in t

def _is_clareamento(txt: str) -> bool:
    return "clareamento" in _norm(txt)

def _is_extracao(txt: str) -> bool:
    t = _norm(txt); return "extração" in t or "extracao" in t

def _is_restauracao(txt: str) -> bool:
    t = _norm(txt); return "restauração" in t or "restauracao" in t

def _has_boca_saudavel(txt: str) -> bool:
    t = _norm(txt); return "boca saudável" in t or "boca saudavel" in t

def _has_presc_forte(txt: str) -> bool:
    t = _norm(txt)
    keys = [
        "amoxicilina", "clavulanato", "metronidazol", "azitromicina", "azitro",
        "antibiotico", "antibiótico",
        "ibuprofeno 600", "diclofenaco", "nimesulida", "corticóide", "corticoide"
    ]
    return any(k in t for k in keys)

TOOTH_RE = re.compile(r"dente\s*(\d{1,2})", flags=re.IGNORECASE)
def extrair_dente(texto: str) -> str:
    m = TOOTH_RE.search(_norm(texto))
    return m.group(1) if m else ""

def add_reason(cur: str, extra: str) -> str:
    cur = (cur or "").strip()
    if not extra: return cur
    if not cur:   return extra
    parts = [p.strip() for p in cur.split(";") if p.strip()]
    if extra not in parts:
        parts.append(extra)
    return "; ".join(parts)

# -------------------------------------------------
# Datas
# -------------------------------------------------
def clamp_years_2022_2025(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normaliza e aceita variações “T”/“Z”
    dt = pd.to_datetime(
        df["created_at"].astype(str).str.replace("T", " ").str.replace("Z", ""),
        errors="coerce"
    )
    def map_year(x):
        if pd.isna(x): return x
        y = min(2025, max(2022, x.year))
        return x.replace(year=y)
    df["created_at"] = dt.map(map_year)
    return df

def shuffle_month_day_preserving_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Embaralha mês/dia por ID (determinístico) preservando ANO e horário.
    Resolve o problema do gráfico “idêntico” entre anos.
    """
    df = df.copy()
    if "id" not in df.columns:  # se não houver id, não mexe
        return df
    def transform(row):
        dt = row["created_at"]
        if pd.isna(dt): return dt
        try:
            rid = int(row["id"])
        except Exception:
            return dt
        new_month = (rid % 12) + 1
        new_day   = ((rid * 7) % 28) + 1  # 1..28
        try:
            return dt.replace(month=new_month, day=new_day)
        except ValueError:
            return dt.replace(month=new_month, day=28)
    df["created_at"] = df.apply(transform, axis=1)
    return df

# -------------------------------------------------
# Motor de regras
# -------------------------------------------------
def aplicar_regras(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # colunas base
    for col, default in [
        ("status", "normal"), ("motivo_atencao", ""), ("executado", ""), ("prescricao", "")
    ]:
        if col not in df.columns:
            df[col] = default

    # tipos
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["status"] = df["status"].fillna("normal").map(_norm)
    df["motivo_atencao"] = df["motivo_atencao"].fillna("")

    # flags auxiliares
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

    # ---------- Regras no MESMO DIA (R5, R9, R10) ----------
    grp_day = df.groupby(["cpf", df["created_at"].dt.date], dropna=True)
    for _, g in grp_day:
        has_profi = g["_is_profilaxia"].any()
        has_canal = g["_is_canal"].any()
        has_extr  = g["_is_extracao"].any()
        has_clare = g["_is_clareamento"].any()

        # R5: incompatíveis no mesmo dia
        reasons: List[str] = []
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
            dups = {k for k, c in counts.items() if c > 1}
            for pos, k in enumerate(key_series.tolist()):
                if k in dups:
                    idx = dup.index[pos]
                    if df.loc[idx, "status"] in LOCKED:
                        continue
                    df.loc[idx, "status"] = "atencao"
                    df.loc[idx, "motivo_atencao"] = add_reason(df.loc[idx, "motivo_atencao"], REGRAS_DESC["duplicidade_mesmo_dia"])

        # R10: prescrição forte sem invasivo
        invasivo = g["_is_canal"].any() or g["_is_extracao"].any() or g["_is_rest"].any()
        if not invasivo and g["_presc_forte"].any():
            for idx in g[g["_presc_forte"]].index:
                if df.loc[idx, "status"] in LOCKED:
                    continue
                df.loc[idx, "status"] = "atencao"
                df.loc[idx, "motivo_atencao"] = add_reason(df.loc[idx, "motivo_atencao"], REGRAS_DESC["presc_forte_sem_invasivo"])

    # ---------- Regras TEMPORAIS por CPF ----------
    for cpf, g in df.groupby("cpf"):
        g = g.sort_values("created_at")

        # pares consecutivos: R1, R3, R4, R8
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            d1, d2 = a["created_at"], b["created_at"]
            if pd.isna(d1) or pd.isna(d2) or d2 < d1:
                continue
            delta = d2 - d1
            reasons: List[str] = []

            if a["_is_profilaxia"] and b["_is_canal"] and delta <= timedelta(days=15):
                reasons.append(REGRAS_DESC["limpeza_canal_15d"])
            if a["_is_clareamento"] and b["_is_canal"] and delta <= timedelta(days=7):
                reasons.append(REGRAS_DESC["clare_canal_7d"])
            if a["_is_rest"] and b["_is_extracao"] and a["_dente"] and b["_dente"] and a["_dente"] == b["_dente"] and delta <= timedelta(days=20):
                reasons.append(REGRAS_DESC["rest_extr_20d"])
            if a["_saudavel"] and (b["_is_canal"] or b["_is_extracao"]) and delta <= timedelta(days=15):
                reasons.append(REGRAS_DESC["saudavel_depois_invasivo"])

            if reasons:
                idx = b.name
                if df.loc[idx, "status"] not in LOCKED:
                    df.loc[idx, "status"] = "atencao"
                for r in reasons:
                    df.loc[idx, "motivo_atencao"] = add_reason(df.loc[idx, "motivo_atencao"], r)

        # R2: canal repetido MESMO dente <= 30d
        canal = g[g["_is_canal"]]
        for dente, gc in canal.groupby("_dente"):
            if not dente:
                continue
            gc = gc.sort_values("created_at")
            for i in range(len(gc) - 1):
                a, b = gc.iloc[i], gc.iloc[i + 1]
                if pd.notna(a["created_at"]) and pd.notna(b["created_at"]) and (b["created_at"] - a["created_at"]) <= timedelta(days=30):
                    idx = b.name
                    if df.loc[idx, "status"] in LOCKED:
                        continue
                    df.loc[idx, "status"] = "atencao"
                    df.loc[idx, "motivo_atencao"] = add_reason(df.loc[idx, "motivo_atencao"], REGRAS_DESC["canal_repetido_30d"])

        # R6: >= 2 canais em 10 dias (dentes quaisquer)
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

        # R7: clareamento repetido <= 14d
        cla = g[g["_is_clareamento"]].sort_values("created_at")
        for i in range(len(cla) - 1):
            a, b = cla.iloc[i], cla.iloc[i + 1]
            if pd.notna(a["created_at"]) and pd.notna(b["created_at"]) and (b["created_at"] - a["created_at"]) <= timedelta(days=14):
                idx = b.name
                if df.loc[idx, "status"] in LOCKED:
                    continue
                df.loc[idx, "status"] = "atencao"
                df.loc[idx, "motivo_atencao"] = add_reason(df.loc[idx, "motivo_atencao"], REGRAS_DESC["clare_rep_14d"])

    # Final: quem é atenção tem motivo obrigatório; quem NÃO é atenção não carrega motivo
    df["status"] = df["status"].replace("", "normal")
    is_atencao = df["status"].astype(str).str.lower().eq("atencao")
    df.loc[~is_atencao, "motivo_atencao"] = ""
    df.loc[is_atencao & (df["motivo_atencao"].astype(str).str.strip() == ""), "motivo_atencao"] = "Motivo não informado (revisar)."

    # limpa auxiliares
    df = df.drop(columns=[
        "_is_profilaxia","_is_canal","_is_clareamento","_is_extracao","_is_rest",
        "_saudavel","_presc_forte","_dente"
    ], errors="ignore")

    return df

# -------------------------------------------------
# main
# -------------------------------------------------
def main():
    # Lê CPF como string (evita perder zeros à esquerda)
    df = pd.read_csv(CSV, dtype={"cpf": str})

    # 1) Normaliza anos e 2) embaralha mês/dia por id (mantém o ANO)
    df = clamp_years_2022_2025(df)
    df = shuffle_month_day_preserving_year(df)

    # 1. Preencher created_at vazio com valor determinístico
    if "id" in df.columns:
        base = pd.Timestamp("2024-01-01 09:00:00")
        df["created_at"] = df["created_at"].fillna(
            df["id"].apply(lambda i: base + pd.Timedelta(days=int(i) % 365))
        )
    else:
        df["created_at"] = df["created_at"].fillna(pd.Timestamp("2024-01-01 09:00:00"))

    # 3) Aplica motor de regras (preenche status/motivo_atencao)
    df = aplicar_regras(df)

    # 2. Remover o legado attention_reason (mescla antes de dropar)
    if "attention_reason" in df.columns:
        df["motivo_atencao"] = df.apply(
            lambda r: (str(r.get("motivo_atencao","")).strip() + (" | " + str(r["attention_reason"]).strip() if str(r["attention_reason"]).strip() else "")).strip(" |"),
            axis=1
        )
        df = df.drop(columns=["attention_reason"])

    # 4) Ordena por data (opcional, melhora leitura)
    if "created_at" in df.columns:
        df = df.sort_values("created_at")

    # 5) Garante tipos coerentes
    if "suspicious" in df.columns:
        df["suspicious"] = df["suspicious"].astype(bool)

    # 3. Ordenar colunas para refletir o que o front espera
    cols_order = [
        "id","paciente_nome","cpf","procedimento","executado","prescricao","dentista",
        "created_at","fraude","status","suspicious","motivos","motivo_atencao"
    ]
    cols = [c for c in cols_order if c in df.columns] + [c for c in df.columns if c not in cols_order]
    df = df[cols]

    df.to_csv(CSV, index=False)
    print("✔ CSV atualizado: datas ok, regras aplicadas e 'motivo_atencao' preenchido para todos os casos em atenção.")

if __name__ == "__main__":
    main()