#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML4N Project 7 — Fase 3b (Sezione 2.2a): tre modelli con i parametri di default, su BoW e TF-IDF.

Modelli (tutti trattati nel corso):
  - Logistic Regression  -> loss logistica (6_EmpiricalRiskMinimization, 9_GradientLearning, lab_5)
  - Linear SVM           -> hinge loss (6_EmpiricalRiskMinimization, 8_Metrics)
  - k-Nearest Neighbors  -> 5_ThreeComponentsML, lab_2
Ognuno con la configurazione di DEFAULT di scikit-learn (nessun parametro passato): il tuning e' il
punto 2.2c. Per kNN il default e' la distanza euclidea (metric='minkowski', p=2).

Per ogni combinazione modello x rappresentazione: training sui 4.000 utenti di training (vettorizzazione
compresa, dentro la Pipeline), valutazione su training E test, confusion matrix e classification report,
tempo di training e warning di convergenza.

Esecuzione, dalla cartella cluster/ (dopo fase3a_dataset.py):
    python fase3b_modelli_default.py
Output in cluster/risultati_fase3/: tables/, figures/, classification_reports_default.txt, report_fase3b.md
"""

from __future__ import annotations

import logging
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from supervised_common import ETICHETTE, RISULTATI, carica_split, crea_pipeline

TAB, FIG = RISULTATI / "tables", RISULTATI / "figures"
log = logging.getLogger("ML4N")

MODELLI = {
    "Logistic Regression": LogisticRegression,
    "Linear SVM": LinearSVC,
    "kNN": KNeighborsClassifier,
}
RAPPRESENTAZIONI = {"BoW": "bow", "TF-IDF": "tfidf"}
NOMI_CLASSI = [ETICHETTE[0], ETICHETTE[1]]


def metriche(y_vero, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_vero, y_pred),
        "precision femminile": precision_score(y_vero, y_pred, pos_label=1, zero_division=0),
        "recall femminile": recall_score(y_vero, y_pred, pos_label=1, zero_division=0),
        "F1 maschile": f1_score(y_vero, y_pred, pos_label=0, zero_division=0),
        "F1 femminile": f1_score(y_vero, y_pred, pos_label=1, zero_division=0),
        "macro-F1": f1_score(y_vero, y_pred, average="macro", zero_division=0),
    }


def addestra_e_valuta(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[str]]:
    righe, confusioni, testi = [], {}, []
    for nome_modello, Classe in MODELLI.items():
        for nome_rapp, rapp in RAPPRESENTAZIONI.items():
            pipe = crea_pipeline(rapp, Classe())          # parametri di default
            with warnings.catch_warnings(record=True) as avvisi:
                warnings.simplefilter("always", ConvergenceWarning)
                t0 = time.perf_counter()
                pipe.fit(train["doc"], train["gender"])
                t_train = time.perf_counter() - t0
            converge = not any(issubclass(a.category, ConvergenceWarning) for a in avvisi)

            for nome_ins, df in (("training", train), ("test", test)):
                t0 = time.perf_counter()
                pred = pipe.predict(df["doc"])
                t_pred = time.perf_counter() - t0
                m = metriche(df["gender"], pred)
                righe.append({"modello": nome_modello, "rappresentazione": nome_rapp, "insieme": nome_ins,
                              **{k: round(v, 4) for k, v in m.items()},
                              "tempo training [s]": round(t_train, 2), "tempo predizione [s]": round(t_pred, 2),
                              "convergenza": "si" if converge else "NO (ConvergenceWarning)"})
                cm = confusion_matrix(df["gender"], pred, labels=[0, 1])
                confusioni[(nome_modello, nome_rapp, nome_ins)] = cm
                cr = classification_report(df["gender"], pred, labels=[0, 1], target_names=NOMI_CLASSI,
                                           digits=3, zero_division=0)
                titolo = f"{nome_modello} — {nome_rapp} — {nome_ins}"
                testi.append(f"{'=' * len(titolo)}\n{titolo}\n{'=' * len(titolo)}\n"
                             f"Confusion matrix (righe = vero, colonne = predetto; ordine: maschile, femminile)\n"
                             f"{pd.DataFrame(cm, index=['vero maschile', 'vero femminile'], columns=['pred. maschile', 'pred. femminile']).to_string()}\n\n"
                             f"{cr}")
                log.info("%s | macro-F1 %.3f | accuracy %.3f", titolo, m["macro-F1"], m["accuracy"])
    return pd.DataFrame(righe), confusioni, testi


def grafico_confusioni(confusioni: dict, insieme: str) -> None:
    fig, axes = plt.subplots(len(MODELLI), len(RAPPRESENTAZIONI), figsize=(8.5, 3.0 * len(MODELLI)))
    for i, nome_modello in enumerate(MODELLI):
        for j, nome_rapp in enumerate(RAPPRESENTAZIONI):
            ax = axes[i, j]
            cm = confusioni[(nome_modello, nome_rapp, insieme)]
            quota = cm / cm.sum(axis=1, keepdims=True)          # quota per classe vera (recall sulla diagonale)
            ax.imshow(quota, cmap="Blues", vmin=0, vmax=1)
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, f"{cm[r, c]}\n({quota[r, c]:.0%})", ha="center", va="center",
                            color="white" if quota[r, c] > 0.6 else "black", fontsize=10)
            ax.set_xticks([0, 1], ["pred. M", "pred. F"]); ax.set_yticks([0, 1], ["vero M", "vero F"])
            ax.set_title(f"{nome_modello} — {nome_rapp}", fontsize=10)
    fig.suptitle(f"Confusion matrix sul {insieme} (parametri di default; % = quota della classe vera)", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG / f"confusion_default_{insieme}.png", dpi=150, bbox_inches="tight")
    plt.close()


def grafico_train_test(tab: pd.DataFrame) -> None:
    piv = tab.pivot_table(index=["modello", "rappresentazione"], columns="insieme", values="macro-F1", sort=False)
    etichette = [f"{m}\n{r}" for m, r in piv.index]
    x = np.arange(len(piv))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - 0.2, piv["training"], width=0.4, color="#9aa5b1", label="training")
    ax.bar(x + 0.2, piv["test"], width=0.4, color="#2f6fd6", label="test")
    for xi, (tr, te) in zip(x, piv[["training", "test"]].to_numpy()):
        ax.text(xi - 0.2, tr + 0.01, f"{tr:.2f}", ha="center", fontsize=9)
        ax.text(xi + 0.2, te + 0.01, f"{te:.2f}", ha="center", fontsize=9)
    ax.axhline(0.422, color="black", ls="--", lw=1, label="baseline (classe maggioritaria) 0,42")
    ax.set_xticks(x, etichette, fontsize=9); ax.set_ylim(0, 1.22); ax.set_ylabel("macro-F1")
    ax.set_title("macro-F1 su training e test — parametri di default")
    ax.legend(loc="upper center", ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "macro_f1_default_train_test.png", dpi=150, bbox_inches="tight")
    plt.close()


def report(tab: pd.DataFrame) -> None:
    def md(df: pd.DataFrame) -> str:
        righe = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
        righe += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
        return "\n".join(righe)

    test = tab[tab["insieme"] == "test"].sort_values("macro-F1", ascending=False)
    migliore = test.iloc[0]
    sintesi = (tab.pivot_table(index=["modello", "rappresentazione"], columns="insieme",
                               values=["accuracy", "macro-F1"], sort=False).round(3))
    sintesi.columns = [f"{m} {i}" for m, i in sintesi.columns]
    sintesi = sintesi.reset_index()
    tempi = tab[tab["insieme"] == "training"][["modello", "rappresentazione", "tempo training [s]", "convergenza"]]
    testo = f"""# ML4N Project 7 — Fase 3b (Sezione 2.2a): modelli con parametri di default

Training sui 4.000 utenti di training, valutazione su training e test (1.000 utenti). Baseline: macro-F1 0,422, accuracy 0,730.

## Sintesi
{md(sintesi)}

Migliore sul test (macro-F1): **{migliore['modello']} con {migliore['rappresentazione']}** — macro-F1 {migliore['macro-F1']:.3f}, accuracy {migliore['accuracy']:.3f}.

## Metriche complete
{md(tab.drop(columns=["tempo predizione [s]"]))}

## Tempi di training e convergenza
{md(tempi)}

Confusion matrix e classification report di ogni combinazione: `classification_reports_default.txt` e `figures/confusion_default_*.png`.
"""
    (RISULTATI / "report_fase3b.md").write_text(testo, encoding="utf-8")


def main() -> int:
    for p in (TAB, FIG):
        p.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(RISULTATI / "esecuzione_fase3b.log", encoding="utf-8")])
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)

    train, test = carica_split()
    log.info("Training: %s utenti — test: %s utenti", f"{len(train):,}", f"{len(test):,}")
    tab, confusioni, testi = addestra_e_valuta(train, test)

    tab.to_csv(TAB / "metriche_default.csv", index=False, encoding="utf-8")
    (RISULTATI / "classification_reports_default.txt").write_text("\n\n".join(testi), encoding="utf-8")
    for ins in ("training", "test"):
        grafico_confusioni(confusioni, ins)
    grafico_train_test(tab)
    report(tab)
    print("\n\n".join(testi))
    log.info("Metriche:\n%s", tab.to_string(index=False))
    log.info("FINE. Report: %s", RISULTATI / "report_fase3b.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Fase 3b interrotta da un errore")
        raise
