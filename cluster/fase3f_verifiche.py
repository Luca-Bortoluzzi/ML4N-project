#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML4N Project 7 — Fase 3f (verifiche sui punti 2.2a e 2.2c): convergenza e tempi misurati senza concorrenza.

Due verifiche chieste dalla revisione del progetto, che NON cambiano la pipeline scelta in fase3d:

A. Convergenza della Logistic Regression di default (punto 2.2a).
   Con i parametri di default lbfgs si ferma a max_iter = 100 senza convergere (ConvergenceWarning). Il
   risultato di default resta quello del punto 2.2a; qui si ripete la stessa configurazione con max_iter alto
   per vedere quante iterazioni servono e se le prestazioni cambiano (9_GradientLearning: criterio di arresto;
   lab_5: dati non standardizzati e convergenza lenta).

B. Tempi e convergenza delle configurazioni candidate (punto 2.2c).
   In fase3d i tempi vengono da GridSearchCV con n_jobs = -1: i fold girano in parallelo e si disturbano.
   Qui ogni configurazione entro una deviazione standard dalla migliore, piu' la migliore di ogni modello,
   viene riaddestrata DA SOLA (n_jobs = 1) sui 4.000 utenti di training, RIPETIZIONI volte, e si misurano:
     - il tempo mediano di addestramento (vettorizzazione compresa);
     - il tempo mediano di predizione sui 4.000 utenti di training (il test non serve e non viene toccato);
       per il kNN e' il costo vero (10_SupervisedTechniques, slide 64; lab_6, punto 2.2.8);
     - il numero di iterazioni dell'ottimizzatore e gli eventuali ConvergenceWarning.
   Il report dice se la configurazione piu' veloce con tempi controllati coincide con quella scelta in fase3d.

Esecuzione, dalla cartella cluster/ (dopo fase3d_tuning.py):
    python fase3f_verifiche.py
Output in cluster/risultati_fase3/: tables/verifica_*.csv, report_fase3f.md
"""

from __future__ import annotations

import logging
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from supervised_common import RISULTATI, carica_split, crea_pipeline

TAB = RISULTATI / "tables"
RIPETIZIONI = 3
MAX_ITER_ALTO = 10_000
log = logging.getLogger("ML4N")


def iterazioni(pipe) -> int | None:
    n = getattr(pipe.named_steps["modello"], "n_iter_", None)
    return None if n is None else int(np.max(n))


def addestra(pipe, train: pd.DataFrame) -> tuple[float, bool]:
    with warnings.catch_warnings(record=True) as avvisi:
        warnings.simplefilter("always", ConvergenceWarning)
        t0 = time.perf_counter()
        pipe.fit(train["doc"], train["gender"])
        t = time.perf_counter() - t0
    return t, not any(issubclass(a.category, ConvergenceWarning) for a in avvisi)


# ----------------------------------------------------------------------------------------------- A
def verifica_logreg_default(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    righe = []
    for rapp, nome_rapp in (("bow", "BoW"), ("tfidf", "TF-IDF")):
        for max_iter in (100, MAX_ITER_ALTO):          # 100 = default di scikit-learn
            pipe = crea_pipeline(rapp, LogisticRegression(max_iter=max_iter))
            t, converge = addestra(pipe, train)
            righe.append({
                "rappresentazione": nome_rapp, "max_iter": max_iter, "iterazioni usate": iterazioni(pipe),
                "converge": "si'" if converge else "no",
                "macro-F1 training": round(f1_score(train["gender"], pipe.predict(train["doc"]), average="macro"), 4),
                "macro-F1 test": round(f1_score(test["gender"], pipe.predict(test["doc"]), average="macro"), 4),
                "tempo training [s]": round(t, 2),
            })
            log.info("LogReg %s max_iter=%d: %s", nome_rapp, max_iter, righe[-1])
    return pd.DataFrame(righe)


# ----------------------------------------------------------------------------------------------- B
def valore(v):
    """I CSV di fase3d salvano None come 'nessuno' e i numeri come testo."""
    if pd.isna(v) or str(v) in ("nessuno", "None", "nan", ""):
        return None
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in str(v) else f
    except ValueError:
        return str(v)


def pipeline_da_riga(r: pd.Series):
    """Ricostruisce la pipeline con gli stessi modelli di base di fase3d_tuning.py."""
    if r["modello"] == "Logistic Regression":
        modello = LogisticRegression(max_iter=3000, C=float(r["C"]), class_weight=valore(r["class_weight"]))
    elif r["modello"] == "Linear SVM":
        modello = LinearSVC(max_iter=5000, C=float(r["C"]), class_weight=valore(r["class_weight"]))
    elif r["modello"] == "kNN":
        modello = KNeighborsClassifier(n_neighbors=int(float(r["n_neighbors"])), metric=str(r["metric"]),
                                       weights=str(r["weights"]))
    else:
        raise ValueError(f"Modello sconosciuto: {r['modello']}")
    pipe = crea_pipeline("tfidf" if r["rappresentazione"] == "TF-IDF" else "bow", modello)
    pipe.set_params(bow__min_df=int(float(r["min_df"])), bow__max_df=float(r["max_df"]))
    return pipe


def verifica_tempi(train: pd.DataFrame) -> pd.DataFrame:
    candidate = pd.read_csv(TAB / "tuning_candidate_entro_1std.csv", keep_default_na=False)
    migliori = pd.read_csv(TAB / "tuning_migliori_per_modello.csv", keep_default_na=False)
    candidate["ruolo"] = ["scelta in fase3d"] + ["candidata"] * (len(candidate) - 1)   # gia' ordinate per tempo
    migliori["ruolo"] = "migliore del modello"
    colonne_param = [c for c in ["rappresentazione", "min_df", "max_df", "C", "class_weight", "n_neighbors",
                                 "metric", "weights"] if c in candidate.columns]
    tutte = pd.concat([candidate, migliori], ignore_index=True)
    tutte = tutte.drop_duplicates(subset=["modello"] + colonne_param, keep="first").reset_index(drop=True)

    righe = []
    for i, r in tutte.iterrows():
        t_fit, t_pred, conv, iters = [], [], True, None
        for _ in range(RIPETIZIONI):
            pipe = pipeline_da_riga(r)
            t, c = addestra(pipe, train)
            t_fit.append(t)
            conv &= c
            iters = iterazioni(pipe)
            t0 = time.perf_counter()
            pipe.predict(train["doc"])
            t_pred.append(time.perf_counter() - t0)
        righe.append({
            "ruolo": r["ruolo"], "modello": r["modello"],
            **{c: r[c] for c in colonne_param},
            "macro-F1 validazione": r["macro-F1 validazione"], "dev. std validazione": r["dev. std validazione"],
            "tempo training fase3d [s]": r["tempo training [s]"],
            "tempo training da solo [s]": round(float(np.median(t_fit)), 2),
            "tempo predizione 4.000 utenti [s]": round(float(np.median(t_pred)), 2),
            "iterazioni": "—" if iters is None else iters,
            "converge": "si'" if conv else "no",
        })
        log.info("%d/%d %s", i + 1, len(tutte), righe[-1])
    return pd.DataFrame(righe)


def md(df: pd.DataFrame) -> str:
    righe = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    righe += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(righe)


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(RISULTATI / "esecuzione_fase3f.log", encoding="utf-8")])
    for f in ("tuning_candidate_entro_1std.csv", "tuning_migliori_per_modello.csv"):
        if not (TAB / f).is_file():
            raise FileNotFoundError(f"Manca {TAB / f}: eseguire prima fase3d_tuning.py")

    train, test = carica_split()
    logreg = verifica_logreg_default(train, test)
    logreg.to_csv(TAB / "verifica_convergenza_logreg_default.csv", index=False)
    tempi = verifica_tempi(train)
    tempi.to_csv(TAB / "verifica_tempi_candidate.csv", index=False)

    cand = tempi[tempi["ruolo"].isin(["scelta in fase3d", "candidata"])]
    piu_veloce = cand.sort_values("tempo training da solo [s]").iloc[0]
    scelta = cand[cand["ruolo"] == "scelta in fase3d"].iloc[0]
    stessa = piu_veloce.name == scelta.name
    conclusione = (
        "Con i tempi misurati da soli la configurazione piu' veloce e' la stessa scelta in fase3d: la scelta e' robusta."
        if stessa else
        f"Con i tempi misurati da soli la piu' veloce sarebbe un'altra ({piu_veloce['modello']}, "
        f"{piu_veloce['rappresentazione']}, {piu_veloce['tempo training da solo [s]']} s contro "
        f"{scelta['tempo training da solo [s]']} s della scelta). Le due sono equivalenti entro una deviazione "
        "standard di validazione: la scelta di fase3d resta valida, ma la regola del tempo minimo e' sensibile "
        "alla misura dei tempi e va presentata come scelta tra configurazioni equivalenti.")
    non_conv = tempi[tempi["converge"] == "no"]

    testo = f"""# ML4N Project 7 — Fase 3f: verifiche di convergenza e tempi (punti 2.2a e 2.2c)

Queste verifiche non cambiano la pipeline scelta in fase3d.

## A. Logistic Regression di default: convergenza
Il risultato richiesto dal punto 2.2a resta quello con i default (max_iter = 100). Qui la stessa
configurazione con max_iter = {MAX_ITER_ALTO:,}.

{md(logreg)}

## B. Configurazioni candidate: tempi misurati da sole e convergenza
{len(cand)} configurazioni entro una deviazione standard dalla migliore, piu' la migliore di ogni modello.
Ogni configurazione e' riaddestrata {RIPETIZIONI} volte da sola (n_jobs = 1) sui 4.000 utenti di training;
tempi mediani. La predizione e' misurata sugli stessi 4.000 utenti di training.

{md(tempi)}

**Conclusione.** {conclusione}

Configurazioni che non convergono: {len(non_conv)}{'' if non_conv.empty else ' (vedi colonna "converge")'}.
"""
    (RISULTATI / "report_fase3f.md").write_text(testo, encoding="utf-8")
    log.info("FINE. Report: %s", RISULTATI / "report_fase3f.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Fase 3f interrotta da un errore")
        raise
