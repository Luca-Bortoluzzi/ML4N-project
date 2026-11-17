#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML4N Project 7 — Fase 3e (Sezione 2.2d): prestazioni per numero di commenti dell'utente.

La pipeline scelta nel punto 2.2c (pipeline_migliore.joblib, gia' addestrata sui 4.000 utenti di training)
viene valutata sugli utenti di TEST divisi per numero di commenti:
  - le fasce della traccia: 1-5, 6-20, 21-100, >100;
  - fasce piu' fini, per capire da quanti commenti in su la classificazione diventa affidabile.
Per ogni fascia: numero di utenti per classe, accuracy, macro-F1, precision/recall/F1 della classe
femminile, confusion matrix, e la baseline (sempre "maschile") calcolata sulla stessa fascia.

Il test qui viene solo usato per descrivere il modello gia' scelto, non per sceglierlo.

Esecuzione, dalla cartella cluster/ (dopo fase3d_tuning.py):
    python fase3e_fasce.py
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

from supervised_common import FASCE, RISULTATI, carica_split

TAB, FIG = RISULTATI / "tables", RISULTATI / "figures"
FILE_MODELLO = RISULTATI / "pipeline_migliore.joblib"
# Fasce piu' fini (limiti superiori inclusi) per vedere l'andamento con maggiore dettaglio
FASCE_FINI = [0, 1, 3, 5, 10, 20, 50, 100, 300, np.inf]
ETICHETTE_FINI = ["1", "2-3", "4-5", "6-10", "11-20", "21-50", "51-100", "101-300", ">300"]
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


def md(df: pd.DataFrame) -> str:
    righe = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    righe += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(righe)


def main() -> int:
    for p in (TAB, FIG):
        p.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(RISULTATI / "esecuzione_fase3e.log", encoding="utf-8")])
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)

    if not FILE_MODELLO.is_file():
        raise FileNotFoundError(f"Manca {FILE_MODELLO}: eseguire prima fase3d_tuning.py")
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

    colonne = ["fascia", "utenti", "maschile", "femminile", "accuracy", "macro-F1", "F1 femminile",
               "recall femminile", "macro-F1 baseline"]
    testo = f"""# ML4N Project 7 — Fase 3e (Sezione 2.2d): prestazioni per numero di commenti

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
cambiando anche pochi utenti; vanno lette insieme al numero di utenti indicato.
"""
    (RISULTATI / "report_fase3e.md").write_text(testo, encoding="utf-8")
    log.info("Fasce della traccia:\n%s", tab.to_string(index=False))
    log.info("Fasce fini:\n%s", tab_fini[colonne].to_string(index=False))
    log.info("Errori per fascia:\n%s", errori.to_string(index=False))
    log.info("FINE. Report: %s", RISULTATI / "report_fase3e.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Fase 3e interrotta da un errore")
        raise
