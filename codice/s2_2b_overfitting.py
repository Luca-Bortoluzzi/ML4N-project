#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML4N Project 7 — Sezione 2.2b: overfitting o underfitting? Le prove.

Strumenti del corso (7_ModelValidationSelection, 8_Metrics_Hyperparameter_tuning):
  1. divario tra metrica sul training e sul test (dai risultati di s2_2a_default.py);
  2. LEARNING CURVE: macro-F1 di training e di validazione al crescere del numero di utenti di training
     -> dice se servirebbero piu' dati (varianza) o se il modello e' troppo semplice (bias);
  3. VALIDATION CURVE: macro-F1 al variare dell'iperparametro che controlla la complessita'
     (C per Logistic Regression e Linear SVM, k per kNN) -> mostra il passaggio underfitting -> overfitting.

Le curve usano SOLO i 4.000 utenti di training, con StratifiedKFold a 5 fold: il test set resta intatto
per la valutazione finale. Ogni fold riaddestra l'intera Pipeline (vocabolario e idf compresi).

Nota: per la Logistic Regression le curve usano max_iter=1000 invece del default (100), cosi' ogni punto
converge e la curva misura l'effetto di C, non quello di un'ottimizzazione interrotta (lab_5).

Esecuzione, dalla cartella codice/ (dopo s2_2a_default.py):
    python s2_2b_overfitting.py
"""

from __future__ import annotations

import logging
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from comune import RISULTATI, SEED, carica_split, crea_pipeline

TAB, FIG = RISULTATI / "tables", RISULTATI / "figures"
N_JOBS = -1                    # tutte le CPU disponibili (sul cluster il calcolo e' molto piu' rapido)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = "f1_macro"
log = logging.getLogger("ML4N")

# Per ogni modello: la rappresentazione migliore nel punto 2.2a, l'iperparametro di complessita' e i valori
# da provare (griglie logaritmiche, 8_Metrics). Verso sinistra il modello e' piu' semplice.
ESPERIMENTI = {
    "Logistic Regression": {
        "rappresentazione": "tfidf", "modello": LogisticRegression(max_iter=1000),
        "parametro": "modello__C", "valori": np.logspace(-4, 2, 7),
        "asse": "C (inverso della regolarizzazione, scala log)", "log": True,
    },
    "Linear SVM": {
        "rappresentazione": "bow", "modello": LinearSVC(),
        "parametro": "modello__C", "valori": np.logspace(-5, 1, 7),
        "asse": "C (inverso della regolarizzazione, scala log)", "log": True,
    },
    "kNN": {
        "rappresentazione": "tfidf", "modello": KNeighborsClassifier(),
        # k grande = modello piu' semplice (decisione mediata su molti vicini): valori in ordine decrescente
        "parametro": "modello__n_neighbors", "valori": np.array([201, 101, 51, 21, 11, 5, 3, 1]),
        "asse": "k = numero di vicini (da semplice a complesso)", "log": False,
    },
}
DIMENSIONI = np.linspace(0.1, 1.0, 7)        # quota dei dati di training di ogni fold


def divario_train_test() -> pd.DataFrame:
    f = TAB / "metriche_default.csv"
    if not f.is_file():
        raise FileNotFoundError(f"Manca {f}: eseguire prima s2_2a_default.py")
    tab = pd.read_csv(f)
    piv = tab.pivot_table(index=["modello", "rappresentazione"], columns="insieme", values="macro-F1", sort=False)
    piv["divario (training - test)"] = piv["training"] - piv["test"]
    piv = piv.round(3).reset_index()
    piv.to_csv(TAB / "overfitting_divario_train_test.csv", index=False)
    return piv


def curve_di_apprendimento(train: pd.DataFrame) -> pd.DataFrame:
    righe = []
    fig, axes = plt.subplots(1, len(ESPERIMENTI), figsize=(15, 4.4), sharey=True)
    for ax, (nome, e) in zip(axes, ESPERIMENTI.items()):
        t0 = time.perf_counter()
        pipe = crea_pipeline(e["rappresentazione"], e["modello"])
        n, s_tr, s_va = learning_curve(pipe, train["doc"], train["gender"], train_sizes=DIMENSIONI, cv=CV,
                                       scoring=SCORING, n_jobs=N_JOBS, shuffle=True, random_state=SEED)
        log.info("Learning curve %s: %.0f s", nome, time.perf_counter() - t0)
        for ni, tr, va in zip(n, s_tr, s_va):
            righe.append({"modello": nome, "rappresentazione": e["rappresentazione"], "utenti di training": int(ni),
                          "macro-F1 training": round(tr.mean(), 4), "dev. std training": round(tr.std(), 4),
                          "macro-F1 validazione": round(va.mean(), 4), "dev. std validazione": round(va.std(), 4)})
        for s, col, et in ((s_tr, "#9aa5b1", "training"), (s_va, "#2f6fd6", "validazione (CV 5 fold)")):
            ax.plot(n, s.mean(axis=1), marker="o", color=col, label=et)
            ax.fill_between(n, s.mean(axis=1) - s.std(axis=1), s.mean(axis=1) + s.std(axis=1), color=col, alpha=0.2)
        ax.axhline(0.422, color="black", ls="--", lw=1, label="baseline 0,42")
        ax.set_title(f"{nome} ({'TF-IDF' if e['rappresentazione'] == 'tfidf' else 'BoW'})")
        ax.set_xlabel("Utenti di training"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1"); axes[0].set_ylim(0.3, 1.05); axes[0].legend(loc="lower right")
    fig.suptitle("Learning curve: il divario tra training e validazione si riduce con piu' dati?")
    plt.tight_layout()
    plt.savefig(FIG / "learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    tab = pd.DataFrame(righe)
    tab.to_csv(TAB / "learning_curve.csv", index=False)
    return tab


def curve_di_validazione(train: pd.DataFrame) -> pd.DataFrame:
    righe = []
    fig, axes = plt.subplots(1, len(ESPERIMENTI), figsize=(15, 4.4), sharey=True)
    for ax, (nome, e) in zip(axes, ESPERIMENTI.items()):
        t0 = time.perf_counter()
        pipe = crea_pipeline(e["rappresentazione"], e["modello"])
        s_tr, s_va = validation_curve(pipe, train["doc"], train["gender"], param_name=e["parametro"],
                                      param_range=e["valori"], cv=CV, scoring=SCORING, n_jobs=N_JOBS)
        log.info("Validation curve %s: %.0f s", nome, time.perf_counter() - t0)
        x = np.arange(len(e["valori"])) if not e["log"] else e["valori"]
        for v, tr, va in zip(e["valori"], s_tr, s_va):
            righe.append({"modello": nome, "parametro": e["parametro"].replace("modello__", ""), "valore": float(v),
                          "macro-F1 training": round(tr.mean(), 4), "macro-F1 validazione": round(va.mean(), 4),
                          "dev. std validazione": round(va.std(), 4),
                          "divario": round(tr.mean() - va.mean(), 4)})
        for s, col, et in ((s_tr, "#9aa5b1", "training"), (s_va, "#2f6fd6", "validazione (CV 5 fold)")):
            ax.plot(x, s.mean(axis=1), marker="o", color=col, label=et)
            ax.fill_between(x, s.mean(axis=1) - s.std(axis=1), s.mean(axis=1) + s.std(axis=1), color=col, alpha=0.2)
        migliore = int(np.argmax(s_va.mean(axis=1)))
        ax.axvline(x[migliore], color="#2f6fd6", ls=":", lw=1)
        ax.annotate(f"migliore: {e['valori'][migliore]:g}\nmacro-F1 {s_va.mean(axis=1)[migliore]:.3f}",
                    (x[migliore], s_va.mean(axis=1)[migliore]), xytext=(6, -32), textcoords="offset points", fontsize=8)
        if e["log"]:
            ax.set_xscale("log")
        else:
            ax.set_xticks(x, [f"{v:g}" for v in e["valori"]])
        ax.axhline(0.422, color="black", ls="--", lw=1, label="baseline 0,42")
        ax.set_title(f"{nome} ({'TF-IDF' if e['rappresentazione'] == 'tfidf' else 'BoW'})")
        ax.set_xlabel(e["asse"]); ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1"); axes[0].set_ylim(0.3, 1.05); axes[0].legend(loc="lower right")
    fig.suptitle("Validation curve: da underfitting (sinistra) a overfitting (destra)")
    plt.tight_layout()
    plt.savefig(FIG / "validation_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    tab = pd.DataFrame(righe)
    tab.to_csv(TAB / "validation_curve.csv", index=False)
    return tab


def report(divario: pd.DataFrame, lc: pd.DataFrame, vc: pd.DataFrame) -> None:
    def md(df: pd.DataFrame) -> str:
        righe = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
        righe += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
        return "\n".join(righe)

    estremi = lc.groupby("modello", sort=False).agg(
        primo=("utenti di training", "first"), val_primo=("macro-F1 validazione", "first"),
        ultimo=("utenti di training", "last"), val_ultimo=("macro-F1 validazione", "last"),
        train_ultimo=("macro-F1 training", "last")).reset_index()
    testo = f"""# ML4N Project 7 — Sezione 2.2b: overfitting e underfitting

## 1. Divario tra training e test (parametri di default)
{md(divario)}

## 2. Learning curve (CV a 5 fold sui 4.000 utenti di training)
{md(estremi.rename(columns={"primo": "utenti (min)", "val_primo": "macro-F1 val. (min)", "ultimo": "utenti (max)",
                             "val_ultimo": "macro-F1 val. (max)", "train_ultimo": "macro-F1 training (max)"}))}

Tabella completa: `tables/learning_curve.csv`; grafico: `figures/learning_curve.png`.

## 3. Validation curve
{md(vc)}

Grafico: `figures/validation_curve.png`.
"""
    (RISULTATI / "report_s2_2b.md").write_text(testo, encoding="utf-8")


def main() -> int:
    for p in (TAB, FIG):
        p.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(RISULTATI / "esecuzione_s2_2b.log", encoding="utf-8")])
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)

    train, _ = carica_split()          # il test non si usa: le curve lavorano solo sul training
    divario = divario_train_test()
    log.info("Divario training/test:\n%s", divario.to_string(index=False))
    lc = curve_di_apprendimento(train)
    log.info("Learning curve:\n%s", lc.to_string(index=False))
    vc = curve_di_validazione(train)
    log.info("Validation curve:\n%s", vc.to_string(index=False))
    report(divario, lc, vc)
    log.info("FINE. Report: %s", RISULTATI / "report_s2_2b.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Sezione 2.2b interrotta da un errore")
        raise
