#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML4N Project 7 — Sezione 2.2d: prestazioni per numero di commenti dell'utente.

La pipeline scelta nel punto 2.2c (pipeline_migliore.joblib, gia' addestrata sui 4.000 utenti di training)
viene valutata sugli utenti di TEST divisi per numero di commenti:
  - le fasce della traccia: 1-5, 6-20, 21-100, >100;
  - fasce piu' fini, per capire da quanti commenti in su la classificazione diventa affidabile.
Per ogni fascia: numero di utenti per classe, accuracy, macro-F1, precision/recall/F1 della classe
femminile, confusion matrix, e la baseline (sempre "maschile") calcolata sulla stessa fascia.

Il test qui viene solo usato per descrivere il modello gia' scelto, non per sceglierlo.

Esperimento di troncamento: le fasce contengono utenti DIVERSI (chi scrive molto puo' anche scrivere di
argomenti diversi), quindi il confronto tra fasce non basta a dire che la causa e' il numero di commenti.
Si prendono gli utenti di test con piu' di 100 commenti e si tengono a caso solo k dei loro commenti
(k = 1, 2, 5, ..., 100), ripetendo il sorteggio piu' volte: gli utenti restano gli stessi e cambia solo la
quantita' di testo. Se la macro-F1 scende come nelle fasce, l'effetto e' dovuto alla quantita' di testo.

Esecuzione, dalla cartella codice/ (dopo s2_2c_tuning.py):
    python s2_2d_fasce.py
"""

from __future__ import annotations

import logging
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score)

from comune import FASCE, FILE_COMMENTI, RISULTATI, SEED, carica_split

TAB, FIG = RISULTATI / "tables", RISULTATI / "figures"
FILE_MODELLO = RISULTATI / "pipeline_migliore.joblib"
# Fasce piu' fini (limiti superiori inclusi) per vedere l'andamento con maggiore dettaglio
FASCE_FINI = [0, 1, 3, 5, 10, 20, 50, 100, 300, np.inf]
ETICHETTE_FINI = ["1", "2-3", "4-5", "6-10", "11-20", "21-50", "51-100", "101-300", ">300"]
# Esperimento di troncamento: utenti di test con piu' di SOGLIA_TRONCAMENTO commenti, k commenti a caso
SOGLIA_TRONCAMENTO = 100
K_TRONCAMENTO = [1, 2, 5, 10, 20, 50, 100]
RIPETIZIONI_TRONCAMENTO = 10
# Per il grafico: punto rappresentativo di ogni fascia fine (utenti diversi) sull'asse dei commenti
CENTRI_FINI = {"1": 1, "2-3": 2.5, "4-5": 4.5, "6-10": 8, "11-20": 15, "21-50": 35, "51-100": 75}
log = logging.getLogger("ML4N")


def metriche_gruppo(df: pd.DataFrame) -> dict:
    y, p = df["gender"], df["predizione"]
    cm = confusion_matrix(y, p, labels=[0, 1])
    baseline = np.zeros(len(y), dtype=int)                  # sempre "maschile"
    return {
        "utenti": len(df), "maschile": int((y == 0).sum()), "femminile": int((y == 1).sum()),
        "accuracy": round(accuracy_score(y, p), 3),
        "macro-F1": round(f1_score(y, p, average="macro", zero_division=0), 3),
        "precision femminile": round(precision_score(y, p, pos_label=1, zero_division=0), 3),
        "recall femminile": round(recall_score(y, p, pos_label=1, zero_division=0), 3),
        "F1 femminile": round(f1_score(y, p, pos_label=1, zero_division=0), 3),
        "F1 maschile": round(f1_score(y, p, pos_label=0, zero_division=0), 3),
        "macro-F1 baseline": round(f1_score(y, baseline, average="macro", zero_division=0), 3),
        "VN (M->M)": int(cm[0, 0]), "FP (M->F)": int(cm[0, 1]), "FN (F->M)": int(cm[1, 0]), "VP (F->F)": int(cm[1, 1]),
    }


def per_fascia(test: pd.DataFrame, colonna: str, ordine: list[str]) -> pd.DataFrame:
    righe = [{"fascia": f, **metriche_gruppo(test[test[colonna] == f])} for f in ordine]
    righe.append({"fascia": "tutti", **metriche_gruppo(test)})
    return pd.DataFrame(righe)


def grafico(tab: pd.DataFrame, tab_fini: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    for ax, t, titolo in ((axes[0], tab, "Fasce della traccia"), (axes[1], tab_fini, "Fasce piu' fini")):
        t = t[t["fascia"] != "tutti"]
        x = np.arange(len(t))
        ax.plot(x, t["macro-F1"], marker="o", color="#2f6fd6", lw=2, label="macro-F1")
        ax.plot(x, t["F1 femminile"], marker="s", color="#2f6fd6", lw=1.2, ls="--", alpha=0.7, label="F1 femminile")
        ax.plot(x, t["macro-F1 baseline"], color="black", ls=":", lw=1, label="baseline (sempre maschile)")
        for xi, (v, n, nf) in enumerate(zip(t["macro-F1"], t["utenti"], t["femminile"])):
            ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
            ax.annotate(f"n={n}\n(F={nf})", (xi, 0.13), ha="center", fontsize=8, color="#555555")
        ax.set_xticks(x, t["fascia"]); ax.set_ylim(0.1, 1.0); ax.grid(alpha=0.3)
        ax.set_xlabel("Commenti per utente"); ax.set_title(titolo)
    axes[0].set_ylabel("Punteggio sul test"); axes[0].legend(loc="center right", fontsize=9)
    fig.suptitle("Prestazioni della pipeline scelta in base al numero di commenti dell'utente (test, 1.000 utenti)")
    plt.tight_layout()
    plt.savefig(FIG / "prestazioni_per_fascia.png", dpi=150, bbox_inches="tight")
    plt.close()


def troncamento(pipe, test: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Stessi utenti, meno commenti: separa l'effetto della quantita' di testo da quello del tipo di utente."""
    utenti = test.loc[test["n_commenti"] > SOGLIA_TRONCAMENTO, ["author", "gender", "doc"]].reset_index(drop=True)
    commenti = pd.read_csv(FILE_COMMENTI, keep_default_na=False, usecols=["author", "clean_text"])
    per_autore = commenti[commenti["author"].isin(set(utenti["author"]))].groupby("author")["clean_text"].apply(list)
    y = utenti["gender"].to_numpy()
    rng = np.random.default_rng(SEED)
    righe = []
    for k in K_TRONCAMENTO:
        punteggi = []
        for _ in range(RIPETIZIONI_TRONCAMENTO):
            documenti = []
            for autore in utenti["author"]:
                testi = per_autore[autore]
                scelti = rng.choice(len(testi), size=min(k, len(testi)), replace=False)
                # stesso documento della Sezione 1: commenti non vuoti uniti da uno spazio
                documenti.append(" ".join(testi[i] for i in scelti if testi[i]))
            pred = pipe.predict(pd.Series(documenti))
            punteggi.append(f1_score(y, pred, average="macro", zero_division=0))
        righe.append({"commenti tenuti": str(k), "macro-F1 media": round(float(np.mean(punteggi)), 3),
                      "dev. std tra i sorteggi": round(float(np.std(punteggi)), 3)})
    pred_tutti = pipe.predict(utenti["doc"])
    righe.append({"commenti tenuti": "tutti", "macro-F1 media": round(f1_score(y, pred_tutti, average="macro"), 3),
                  "dev. std tra i sorteggi": 0.0})
    return pd.DataFrame(righe), len(utenti), int(y.sum())


def grafico_troncamento(tronc: pd.DataFrame, tab_fini: pd.DataFrame) -> None:
    t = tronc[tronc["commenti tenuti"] != "tutti"]
    x = t["commenti tenuti"].astype(int).to_numpy()
    media, sd = t["macro-F1 media"].to_numpy(), t["dev. std tra i sorteggi"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(x, media - sd, media + sd, color="#2f6fd6", alpha=0.15, lw=0)
    ax.plot(x, media, marker="o", color="#2f6fd6", lw=2,
            label=f"stessi utenti (>{SOGLIA_TRONCAMENTO} commenti), k commenti a caso")
    fini = tab_fini[tab_fini["fascia"].isin(list(CENTRI_FINI))]
    ax.plot([CENTRI_FINI[f] for f in fini["fascia"]], fini["macro-F1"], marker="s", ls="--", color="#eb6834",
            lw=1.2, label="utenti diversi, per fascia di commenti")
    ax.set_xscale("log"); ax.set_xticks(x, [str(v) for v in x]); ax.set_ylim(0.3, 1.0); ax.grid(alpha=0.3)
    ax.set_xlabel("Commenti per utente (scala logaritmica)"); ax.set_ylabel("macro-F1 sul test")
    ax.set_title("Quantita' di testo o tipo di utente? Esperimento di troncamento")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "troncamento_commenti.png", dpi=150, bbox_inches="tight")
    plt.close()


def md(df: pd.DataFrame) -> str:
    righe = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    righe += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(righe)


def main() -> int:
    for p in (TAB, FIG):
        p.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(RISULTATI / "esecuzione_s2_2d.log", encoding="utf-8")])
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)

    if not FILE_MODELLO.is_file():
        raise FileNotFoundError(f"Manca {FILE_MODELLO}: eseguire prima s2_2c_tuning.py")
    pipe = joblib.load(FILE_MODELLO)                 # gia' addestrata sui 4.000 utenti di training
    log.info("Pipeline caricata: %s", pipe.named_steps["modello"])

    _, test = carica_split()
    test["predizione"] = pipe.predict(test["doc"])
    test["fascia_fine"] = pd.cut(test["n_commenti"], bins=FASCE_FINI, labels=ETICHETTE_FINI).astype(str)

    tab = per_fascia(test.assign(fascia=test["fascia_commenti"].astype(str)), "fascia", FASCE)
    tab_fini = per_fascia(test, "fascia_fine", ETICHETTE_FINI)
    tab.to_csv(TAB / "prestazioni_per_fascia.csv", index=False)
    tab_fini.to_csv(TAB / "prestazioni_per_fascia_fine.csv", index=False)
    grafico(tab, tab_fini)

    # Errori: in quale fascia cadono gli utenti classificati male?
    errori = (test.assign(errore=test["gender"] != test["predizione"])
              .groupby("fascia_commenti", observed=False)["errore"].agg(["sum", "mean"])
              .rename(columns={"sum": "utenti sbagliati", "mean": "quota sbagliata"}).round(3)
              .reset_index().rename(columns={"fascia_commenti": "fascia"}))
    errori["quota sul totale degli errori"] = (errori["utenti sbagliati"] / errori["utenti sbagliati"].sum()).round(3)
    errori.to_csv(TAB / "errori_per_fascia.csv", index=False)

    tronc, n_tronc, f_tronc = troncamento(pipe, test)
    tronc.to_csv(TAB / "troncamento_commenti.csv", index=False)
    grafico_troncamento(tronc, tab_fini)

    colonne = ["fascia", "utenti", "maschile", "femminile", "accuracy", "macro-F1", "F1 femminile",
               "recall femminile", "macro-F1 baseline"]
    testo = f"""# ML4N Project 7 — Sezione 2.2d: prestazioni per numero di commenti

Pipeline valutata: quella scelta nel punto 2.2c (`pipeline_migliore.joblib`), addestrata sui 4.000 utenti
di training. Gruppi formati dai 1.000 utenti di test in base al numero di commenti.

## Fasce della traccia
{md(tab[colonne])}

## Fasce più fini
{md(tab_fini[colonne])}

## Dove cadono gli errori
{md(errori)}

Confusion matrix per fascia (VN, FP, FN, VP): `tables/prestazioni_per_fascia.csv`. Grafico: `figures/prestazioni_per_fascia.png`.

Attenzione: nelle fasce con pochi utenti (in particolare la classe femminile) le metriche variano molto
cambiando anche pochi utenti; vanno lette insieme al numero di utenti indicato. Le fasce contengono anche
utenti diversi: il confronto tra fasce da solo non dimostra che la causa sia il numero di commenti.

## Esperimento di troncamento (stessi utenti, meno commenti)
{n_tronc} utenti di test con più di {SOGLIA_TRONCAMENTO} commenti ({f_tronc} femminili); per ogni k, {RIPETIZIONI_TRONCAMENTO} sorteggi
di k commenti per utente. La pipeline non viene riaddestrata.

{md(tronc)}

Se la macro-F1 degli stessi utenti scende quando si tolgono commenti, l'aumento osservato tra le fasce è
dovuto (almeno in parte) alla quantità di testo e non solo al tipo di utente. Grafico: `figures/troncamento_commenti.png`.
"""
    (RISULTATI / "report_s2_2d.md").write_text(testo, encoding="utf-8")
    log.info("Fasce della traccia:\n%s", tab.to_string(index=False))
    log.info("Fasce fini:\n%s", tab_fini[colonne].to_string(index=False))
    log.info("Errori per fascia:\n%s", errori.to_string(index=False))
    log.info("Troncamento (%d utenti, %d femminili):\n%s", n_tronc, f_tronc, tronc.to_string(index=False))
    log.info("FINE. Report: %s", RISULTATI / "report_s2_2d.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Sezione 2.2d interrotta da un errore")
        raise
