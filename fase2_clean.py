#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML4N Project 7 — Fase 2 (passi 1-2): pulizia e preprocessing del testo.

Legge i CSV ORIGINALI (gli stessi della Fase 1), ripete la pulizia di qualita'
dei dati della Fase 1 e applica preprocessing.preprocess() a ogni commento.

Esecuzione (sul cluster, dalla cartella dello script):
    source ~/ml4n-env/bin/activate
    python fase2_clean.py                                   # prova veloce (vedi N_RIGHE_TEST)
    nohup python fase2_clean.py > log_fase2.txt 2>&1 &      # dati completi

Output in CARTELLA_RISULTATI:
    supervised_preproc.csv, unsupervised_preproc.csv  -> dataset con colonne clean_text e n_token
                                                         (n_token = 0: commento vuoto dopo il preprocessing)
    fase2_riepilogo.csv                               -> conteggi da riportare nella relazione
    fase2_esempi_<dataset>.csv                        -> 30 esempi prima/dopo da controllare a mano
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from preprocessing import SEGNAPOSTO_RIMOSSI, preprocess_many

# ======================================================================
# CONFIGURAZIONE — modificare solo qui
# ======================================================================
CARTELLA_DATI = Path(__file__).resolve().parent           # dove stanno i CSV originali
FILE_SUPERVISED = CARTELLA_DATI / "data_supervised.csv"
FILE_UNSUPERVISED = CARTELLA_DATI / "data_unsupervised.csv"
CARTELLA_RISULTATI = CARTELLA_DATI / "risultati_fase2"

N_RIGHE_TEST = 1000      # prova: legge solo le prime N righe. None = dataset completo
N_JOBS = 16              # processi paralleli (il cluster ne ha molti; 1 = nessun parallelismo)
DIMENSIONE_BLOCCO = 20_000
SEED = 42
COLONNE = ["author", "subreddit", "created_utc", "body"]


def leggi_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"File non trovato: {path}. Correggere CARTELLA_DATI / FILE_*.")
    df = pd.read_csv(path, nrows=N_RIGHE_TEST, low_memory=False, encoding="utf-8")
    df.columns = df.columns.str.strip()
    mancanti = set(COLONNE) - set(df.columns)
    if mancanti:
        raise ValueError(f"{path.name}: mancano le colonne {sorted(mancanti)}")
    return df[COLONNE]


def pulizia_qualita(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Stessa politica della Fase 1 (slide 'Data Quality') + segnaposto rimossi."""
    stats = {"righe_lette": len(df)}
    df = df.drop_duplicates(keep="first").copy()
    stats["duplicati_esatti_rimossi"] = stats["righe_lette"] - len(df)

    df["author"] = df["author"].astype("string").str.strip()
    df["subreddit"] = df["subreddit"].astype("string").str.strip()
    df["body"] = df["body"].astype("string")
    vuoti = df["author"].isna() | (df["author"] == "") | df["body"].isna() | (df["body"].str.strip() == "")
    stats["righe_senza_testo_o_autore"] = int(vuoti.sum())
    df = df.loc[~vuoti]

    segnaposto = df["body"].str.strip().str.lower().isin(SEGNAPOSTO_RIMOSSI)
    stats["commenti_deleted_removed"] = int(segnaposto.sum())
    df = df.loc[~segnaposto].copy()
    return df, stats


def preprocess_parallelo(testi: list[str]) -> list[str]:
    blocchi = [testi[i:i + DIMENSIONE_BLOCCO] for i in range(0, len(testi), DIMENSIONE_BLOCCO)]
    if N_JOBS == 1 or len(blocchi) == 1:
        risultati = [preprocess_many(b) for b in blocchi]
    else:
        risultati = Parallel(n_jobs=N_JOBS)(delayed(preprocess_many)(b) for b in blocchi)
    return [t for blocco in risultati for t in blocco]


def elabora(nome: str, path: Path, out: Path) -> dict:
    log = logging.getLogger("ML4N")
    t0 = time.perf_counter()
    df, stats = pulizia_qualita(leggi_csv(path))
    log.info("%s: %s righe lette, %s dopo la pulizia di qualita'", nome,
             f"{stats['righe_lette']:,}", f"{len(df):,}")

    df["clean_text"] = preprocess_parallelo(df["body"].tolist())
    df["n_token"] = df["clean_text"].str.split().str.len().fillna(0).astype("int32")

    # I commenti rimasti senza token (solo stopword, numeri, link...) NON vengono
    # eliminati qui: alcuni utenti supervised hanno solo commenti di questo tipo e
    # perderemmo la loro etichetta. Restano con n_token = 0; ogni fase decide:
    #  - Sezione 2: l'aggregazione per autore li tiene (documento eventualmente vuoto)
    #  - Sezione 3: il clustering applichera' una soglia minima di token
    vuoti_dopo = df["n_token"] == 0
    stats["commenti_vuoti_dopo_preprocessing"] = int(vuoti_dopo.sum())
    autori_tutti = set(df["author"])
    autori_con_testo = set(df.loc[~vuoti_dopo, "author"])
    stats["autori_solo_commenti_vuoti"] = len(autori_tutti - autori_con_testo)

    tutti_i_token = df.loc[~vuoti_dopo, "clean_text"].str.split().explode()
    stats.update({
        "dataset": nome,
        "commenti_finali": len(df),
        "autori_finali": int(df["author"].nunique()),
        "token_totali": int(df["n_token"].sum()),
        "vocabolario_stem_distinti": int(tutti_i_token.nunique()),
        "token_per_commento_media": round(float(df.loc[~vuoti_dopo, "n_token"].mean()), 2),
        "token_per_commento_mediana": float(df.loc[~vuoti_dopo, "n_token"].median()),
        "secondi_elaborazione": round(time.perf_counter() - t0, 1),
    })

    esempi = df.sample(n=min(30, len(df)), random_state=SEED)[["body", "clean_text"]]
    esempi.to_csv(out / f"fase2_esempi_{nome}.csv", index=False, encoding="utf-8")
    df[COLONNE + ["clean_text", "n_token"]].to_csv(out / f"{nome}_preproc.csv", index=False, encoding="utf-8")
    log.info("%s: %s commenti salvati, di cui %s vuoti dopo il preprocessing (%s autori con soli commenti vuoti), "
             "vocabolario %s stem (%.0f s)",
             nome, f"{stats['commenti_finali']:,}", f"{stats['commenti_vuoti_dopo_preprocessing']:,}",
             f"{stats['autori_solo_commenti_vuoti']:,}",
             f"{stats['vocabolario_stem_distinti']:,}", stats["secondi_elaborazione"])
    return stats


def main() -> int:
    out = CARTELLA_RISULTATI.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "esecuzione_fase2.log", encoding="utf-8")])
    log = logging.getLogger("ML4N")
    modo = f"PROVA su {N_RIGHE_TEST} righe" if N_RIGHE_TEST else "DATASET COMPLETO"
    log.info("Fase 2 — %s — risultati in %s", modo, out)

    riepilogo = [elabora("supervised", FILE_SUPERVISED, out),
                 elabora("unsupervised", FILE_UNSUPERVISED, out)]
    cols = ["dataset", "righe_lette", "duplicati_esatti_rimossi", "righe_senza_testo_o_autore",
            "commenti_deleted_removed", "commenti_finali", "commenti_vuoti_dopo_preprocessing",
            "autori_solo_commenti_vuoti",
            "autori_finali", "token_totali", "vocabolario_stem_distinti",
            "token_per_commento_media", "token_per_commento_mediana", "secondi_elaborazione"]
    tab = pd.DataFrame(riepilogo)[cols]
    tab.to_csv(out / "fase2_riepilogo.csv", index=False, encoding="utf-8")
    log.info("Riepilogo:\n%s", tab.T.to_string(header=False))
    log.info("FINE.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Fase 2 interrotta da un errore")
        raise
