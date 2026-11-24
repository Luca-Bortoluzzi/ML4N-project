#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML4N Project 7 — Fase 3d (Sezione 2.2c): tuning degli iperparametri con cross-validation.

Procedura (8_Metrics_Hyperparameter_tuning, 7_ModelValidationSelection):
  - GridSearchCV sull'INTERA pipeline (vettorizzazione + modello), con StratifiedKFold a 5 fold sui soli
    4.000 utenti di training: il test set non viene mai usato per scegliere;
  - si ottimizzano anche gli iperparametri del PREPROCESSING (slide 35): rappresentazione (BoW / TF-IDF),
    MIN_DF, MAX_DF; per i modelli: C e class_weight (LogReg, SVM), k e la distanza (kNN);
  - griglie in scala logaritmica (slide 36);
  - metrica: macro-F1 (classi 73/27); si registrano anche accuracy, punteggio di training e tempo di training.

Scelta finale: la pipeline con la macro-F1 media di validazione piu' alta; tra le configurazioni la cui
media cade entro una deviazione standard (tra i fold) dalla migliore, si sceglie quella con il tempo di
training minore. E' una regola EURISTICA, non un test statistico: la deviazione standard tra i fold descrive la
variabilita' ma non e' un intervallo di confidenza (7_ModelValidationSelection, slide 41). I tempi vengono da
GridSearchCV con n_jobs = -1 e sono disturbati dai fold in parallelo: fase3f_verifiche.py li rimisura da soli.
Solo la pipeline scelta viene valutata sul test, una volta.

Esecuzione, dalla cartella cluster/ (dopo fase3a_dataset.py):
    python fase3d_tuning.py                          # griglia completa (consigliato sul cluster)
    ML4N_GRIGLIA=ridotta python fase3d_tuning.py     # prova veloce con pochi valori
"""

from __future__ import annotations

import logging
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from supervised_common import ETICHETTE, RISULTATI, SEED, TfidfSlide, carica_split, crea_pipeline

TAB, FIG = RISULTATI / "tables", RISULTATI / "figures"
FILE_MODELLO = RISULTATI / "pipeline_migliore.joblib"
N_JOBS = -1
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
RIDOTTA = os.environ.get("ML4N_GRIGLIA", "").strip().lower() == "ridotta"
log = logging.getLogger("ML4N")

# Rappresentazione come iperparametro: con 'passthrough' il passo TF-IDF viene saltato e resta la BoW.
RAPPRESENTAZIONE = [TfidfSlide(), "passthrough"]
PREPROCESSING = {"bow__min_df": [5, 10, 20], "bow__max_df": [0.5, 0.7, 1.0], "tfidf": RAPPRESENTAZIONE}
if RIDOTTA:
    PREPROCESSING = {"bow__min_df": [10], "bow__max_df": [0.7], "tfidf": RAPPRESENTAZIONE}

GRIGLIE = {
    "Logistic Regression": {
        # max_iter alto: con i conteggi non standardizzati lbfgs converge lentamente (lab_5)
        "modello": LogisticRegression(max_iter=3000),
        # griglia estesa verso C piccoli: una prima ricerca su [0.001, 1] aveva il migliore sul bordo (0.001)
        "griglia": {**PREPROCESSING, "modello__C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] if not RIDOTTA else [1e-4, 1e-2],
                    "modello__class_weight": [None, "balanced"]},
    },
    "Linear SVM": {
        "modello": LinearSVC(max_iter=5000),
        # griglia estesa verso C piccoli: una prima ricerca su [0.0001, 1] aveva il migliore sul bordo (0.0001)
        "griglia": {**PREPROCESSING, "modello__C": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] if not RIDOTTA else [1e-5, 1e-3],
                    "modello__class_weight": [None, "balanced"]},
    },
    "kNN": {
        "modello": KNeighborsClassifier(),
        # coseno: confronta la direzione dei vettori (cosa scrive l'utente), non la lunghezza (quanto scrive)
        "griglia": {**{k: v for k, v in PREPROCESSING.items() if k != "bow__max_df"}, "bow__max_df": [0.7],
                    # estesa verso k grandi: una prima ricerca su [5, 51] aveva il migliore sul bordo (51)
                    "modello__n_neighbors": [11, 21, 51, 101] if not RIDOTTA else [51],
                    "modello__metric": ["euclidean", "cosine"],
                    "modello__weights": ["uniform", "distance"] if not RIDOTTA else ["uniform"]},
    },
}


def nome_rappresentazione(v) -> str:
    return "TF-IDF" if isinstance(v, TfidfSlide) else "BoW"


def esegui_ricerche(train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    risultati, ricerche = [], {}
    for nome, g in GRIGLIE.items():
        n_comb = int(np.prod([len(v) for v in g["griglia"].values()]))
        log.info("%s: %d combinazioni x 5 fold = %d addestramenti", nome, n_comb, 5 * n_comb)
        t0 = time.perf_counter()
        ricerca = GridSearchCV(crea_pipeline("tfidf", g["modello"]), g["griglia"], cv=CV,
                               scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"}, refit="macro_f1",
                               return_train_score=True, n_jobs=N_JOBS, error_score="raise")
        ricerca.fit(train["doc"], train["gender"])
        log.info("%s: %.0f s — migliore macro-F1 CV %.4f con %s", nome, time.perf_counter() - t0,
                 ricerca.best_score_, {k: (nome_rappresentazione(v) if k == "tfidf" else v)
                                       for k, v in ricerca.best_params_.items()})
        ricerche[nome] = ricerca

        cv = pd.DataFrame(ricerca.cv_results_)
        tab = pd.DataFrame({
            "modello": nome,
            "rappresentazione": cv["param_tfidf"].map(nome_rappresentazione),
            # None -> "nessuno": con astype(str) resterebbe un valore mancante, e "None" nei CSV viene riletto come NaN
            **{c.replace("param_", "").replace("bow__", "").replace("modello__", ""):
               cv[c].map(lambda v: "nessuno" if v is None else str(v))
               for c in cv.columns if c.startswith("param_") and c != "param_tfidf"},
            "macro-F1 validazione": cv["mean_test_macro_f1"].round(4),
            "dev. std validazione": cv["std_test_macro_f1"].round(4),
            "macro-F1 training": cv["mean_train_macro_f1"].round(4),
            "accuracy validazione": cv["mean_test_accuracy"].round(4),
            "tempo training [s]": cv["mean_fit_time"].round(2),
            "indice": np.arange(len(cv)),
        }).sort_values("macro-F1 validazione", ascending=False)
        tab.to_csv(TAB / f"tuning_cv_{nome.lower().replace(' ', '_')}.csv", index=False)
        risultati.append(tab)
    return pd.concat(risultati, ignore_index=True), ricerche


def scegli(tutti: pd.DataFrame) -> pd.Series:
    """Migliore media; tra quelle entro 1 dev. std dalla migliore, la meno costosa da addestrare."""
    migliore = tutti.sort_values("macro-F1 validazione", ascending=False).iloc[0]
    soglia = migliore["macro-F1 validazione"] - migliore["dev. std validazione"]
    candidate = tutti[tutti["macro-F1 validazione"] >= soglia].sort_values("tempo training [s]")
    candidate.to_csv(TAB / "tuning_candidate_entro_1std.csv", index=False)
    log.info("Migliore assoluta: %s %s, macro-F1 %.4f ± %.4f; %d configurazioni entro 1 dev. std",
             migliore["modello"], migliore["rappresentazione"], migliore["macro-F1 validazione"],
             migliore["dev. std validazione"], len(candidate))
    return candidate.iloc[0]


def pipeline_da_riga(riga: pd.Series, ricerche: dict):
    ricerca = ricerche[riga["modello"]]
    params = ricerca.cv_results_["params"][int(riga["indice"])]
    pipe = clone(ricerca.estimator).set_params(**params)
    return pipe, params


def migliori_per_modello(tutti: pd.DataFrame) -> pd.DataFrame:
    righe = tutti.sort_values("macro-F1 validazione", ascending=False).groupby("modello", sort=False).head(1)
    righe = righe.drop(columns="indice")
    righe.to_csv(TAB / "tuning_migliori_per_modello.csv", index=False)
    return righe


def heatmap(tutti: pd.DataFrame, modello: str, riga_x: str, col_y: str) -> None:
    """Macro-F1 di validazione per due iperparametri, con gli altri fissati al loro valore migliore."""
    d = tutti[tutti["modello"] == modello]
    migliore = d.iloc[0]
    fissi = [c for c in ["rappresentazione", "min_df", "max_df", "C", "class_weight", "n_neighbors", "metric", "weights"]
             if c in d.columns and c not in (riga_x, col_y) and d[c].notna().all()]
    for c in fissi:
        d = d[d[c] == migliore[c]]
    piv = d.pivot_table(index=col_y, columns=riga_x, values="macro-F1 validazione")
    if piv.shape[0] < 2 or piv.shape[1] < 2:
        return
    for asse in ("index", "columns"):
        try:
            ordine = sorted(getattr(piv, asse), key=float)
            piv = piv.reindex(ordine) if asse == "index" else piv.reindex(columns=ordine)
        except ValueError:
            pass
    fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 2.5, 0.8 * piv.shape[0] + 1.8))
    im = ax.imshow(piv.to_numpy(), cmap="Blues", aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iat[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if piv.iat[i, j] > piv.to_numpy().mean() else "black")
    ax.set_xticks(range(piv.shape[1]), piv.columns); ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set_xlabel(riga_x); ax.set_ylabel(col_y)
    ax.set_title(f"{modello}: macro-F1 di validazione (CV 5 fold)\naltri iperparametri: "
                 + ", ".join(f"{c}={migliore[c]}" for c in fissi), fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIG / f"tuning_heatmap_{modello.lower().replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.close()


def valuta_test(nome: str, pipe, train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, str, np.ndarray]:
    t0 = time.perf_counter()
    pipe.fit(train["doc"], train["gender"])
    t_fit = time.perf_counter() - t0
    pred_tr, pred_te = pipe.predict(train["doc"]), pipe.predict(test["doc"])
    m = {"pipeline": nome, "macro-F1 training": round(f1_score(train["gender"], pred_tr, average="macro"), 4),
         "macro-F1 test": round(f1_score(test["gender"], pred_te, average="macro"), 4),
         "accuracy test": round(accuracy_score(test["gender"], pred_te), 4),
         "F1 femminile test": round(f1_score(test["gender"], pred_te, pos_label=1), 4),
         "tempo training [s]": round(t_fit, 2)}
    cr = classification_report(test["gender"], pred_te, labels=[0, 1], target_names=list(ETICHETTE.values()), digits=3)
    return m, cr, confusion_matrix(test["gender"], pred_te, labels=[0, 1])


def md(df: pd.DataFrame) -> str:
    def cella(v) -> str:
        return "—" if (isinstance(v, float) and np.isnan(v)) or str(v) == "nan" else str(v)
    righe = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    righe += ["| " + " | ".join(cella(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(righe)


def main() -> int:
    for p in (TAB, FIG):
        p.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(RISULTATI / "esecuzione_fase3d.log", encoding="utf-8")])
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)
    log.info("Griglia %s", "RIDOTTA (prova)" if RIDOTTA else "completa")

    train, test = carica_split()
    tutti, ricerche = esegui_ricerche(train)
    migliori = migliori_per_modello(tutti)
    log.info("Migliori configurazioni per modello (CV):\n%s", migliori.to_string(index=False))

    for modello, x, y in (("Logistic Regression", "C", "min_df"), ("Linear SVM", "C", "min_df"),
                          ("kNN", "n_neighbors", "metric")):
        heatmap(tutti, modello, x, y)

    scelta = scegli(tutti)
    pipe_scelta, params = pipeline_da_riga(scelta, ricerche)
    m_scelta, cr_scelta, cm_scelta = valuta_test(f"{scelta['modello']} ({scelta['rappresentazione']}) — SCELTA",
                                                 pipe_scelta, train, test)
    joblib.dump(pipe_scelta, FILE_MODELLO)
    log.info("Pipeline scelta: %s\nTest:\n%s\n%s", params, cr_scelta, cm_scelta)

    # Solo a titolo informativo (NON usato per la scelta): test delle migliori configurazioni degli altri modelli
    informativi = []
    for _, riga in migliori.iterrows():
        if riga["modello"] == scelta["modello"]:
            continue
        ricerca = ricerche[riga["modello"]]
        m, _, _ = valuta_test(f"{riga['modello']} ({riga['rappresentazione']}) — informativo",
                              ricerca.best_estimator_, train, test)
        informativi.append(m)
    finale = pd.DataFrame([m_scelta] + informativi)
    finale.to_csv(TAB / "tuning_test_finale.csv", index=False)

    default = pd.read_csv(TAB / "metriche_default.csv") if (TAB / "metriche_default.csv").is_file() else None
    confronto = ""
    if default is not None:
        d = default[(default["modello"] == scelta["modello"]) & (default["rappresentazione"] == scelta["rappresentazione"])
                    & (default["insieme"] == "test")]
        if len(d):
            confronto = (f"Con i parametri di default la stessa combinazione aveva macro-F1 test "
                         f"{d['macro-F1'].iloc[0]:.3f} (punto 2.2a).")

    cm_df = pd.DataFrame(cm_scelta, index=["vero maschile", "vero femminile"], columns=["pred. maschile", "pred. femminile"])
    testo = f"""# ML4N Project 7 — Fase 3d (Sezione 2.2c): tuning con cross-validation

Griglia: {"RIDOTTA (prova)" if RIDOTTA else "completa"}. CV stratificata a 5 fold sui 4.000 utenti di training, metrica macro-F1.

## Migliore configurazione di ogni modello (validazione)
{md(migliori)}

## Scelta
Regola (euristica): macro-F1 di validazione massima; tra le configurazioni entro una deviazione standard dalla
migliore, la piu' veloce da addestrare (vedi `tables/tuning_candidate_entro_1std.csv`). La deviazione standard tra
i fold non e' un intervallo di confidenza; i tempi con n_jobs = -1 sono rimisurati da soli in `report_fase3f.md`.

**Pipeline scelta: {scelta['modello']} con {scelta['rappresentazione']}** — macro-F1 CV {scelta['macro-F1 validazione']:.4f} ± {scelta['dev. std validazione']:.4f}, tempo di training {scelta['tempo training [s]']} s.
Parametri: `{ {k: (nome_rappresentazione(v) if k == 'tfidf' else v) for k, v in params.items()} }`

## Valutazione finale sul test (una sola volta)
{md(finale)}

Le righe "informativo" non sono state usate per la scelta. {confronto}

Confusion matrix della pipeline scelta (test):

{md(cm_df.reset_index().rename(columns={"index": ""}))}

```
{cr_scelta}
```
"""
    (RISULTATI / "report_fase3d.md").write_text(testo, encoding="utf-8")
    log.info("Test finale:\n%s", finale.to_string(index=False))
    log.info("FINE. Report: %s — pipeline salvata in %s", RISULTATI / "report_fase3d.md", FILE_MODELLO)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Fase 3d interrotta da un errore")
        raise
