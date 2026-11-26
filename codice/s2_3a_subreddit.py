#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML4N Project 7 — Sezione 2.3a: shortcut analysis con le feature dei subreddit.

Domanda della traccia: alcuni subreddit sono molto legati a un genere (AskMen, AskWomen, MakeupAddiction):
bastano i subreddit in cui un utente scrive per indovinarne il genere?

Feature: per ogni utente, un vettore con una colonna per subreddit; il valore e' la FRAZIONE dei suoi
commenti scritti in quel subreddit (numero di commenti nel subreddit / numero totale di commenti).

Confronto, con lo stesso split 80/20 e la stessa metrica (macro-F1) della Sezione 2:
  A. solo testo:          la pipeline scelta nel punto 2.2c (pipeline_migliore.joblib, iperparametri invariati);
  B. solo subreddit:      lo stesso modello (Linear SVM, class_weight = balanced) sulle frazioni dei subreddit;
  C. testo + subreddit:   le due matrici affiancate, poi lo stesso modello.

Riferimenti alla teoria (S = pacco di slide, pagina del PDF; Note = note del corso):
  - Pipeline: tutto cio' che impara dai dati (elenco dei subreddit, vocabolario, idf, scala, modello) e'
    addestrato solo sul training, e dentro la CV solo sui fold di training [S7 p. 34, 44];
  - soglia min_utenti sui subreddit = selezione delle feature irrilevanti, come min_df per le parole [S3 p. 47];
  - iperparametri del modello E del preprocessing (min_utenti, scala) scelti insieme con grid search, scala
    logaritmica per C [S8 p. 35, 38], CV a 5 fold stratificata [S7 p. 37-41, 47; Note p. 54], metrica
    macro-F1 per le classi sbilanciate [S8 p. 20; Note p. 55];
  - scelta: configurazione con la macro-F1 di validazione piu' alta [S7 p. 50, 54]; poi riaddestramento sul
    training completo e una sola valutazione sul test [S7 p. 53-54; Note p. 146];
  - scala: per un modello lineare regolarizzato le feature vanno standardizzate, altrimenti la penalita' sui
    pesi dipende dalle unita' di misura [S10 p. 12]; serve quando le scale sono diverse [S3 p. 25, 59], come in
    "testo + subreddit" (TF-IDF fino a centinaia, frazioni in [0, 1]). Si usa lo StandardScaler dei lab
    [Note p. 26] senza sottrarre la media (with_mean=False): sottrarla renderebbe densa la matrice sparsa, e per
    un modello lineare con intercetta lo spostamento finisce nell'intercetta, perche'
    w.(x - mu)/sigma + b = w.x/sigma + (b - w.mu/sigma) [Note p. 47] (liblinear regolarizza anche l'intercetta,
    quindi l'equivalenza e' quasi esatta, non esatta). La normalizzazione min-max (MaxAbsScaler) e' scartata perche' sensibile agli outlier
    [S3 p. 56]: qui il massimo di molte parole viene da un solo utente con 19.534 commenti;
  - per "solo subreddit" le frazioni hanno gia' la stessa unita' di misura: standardizzare o no e' un
    iperparametro del preprocessing, scelto in CV [S8 p. 35];
  - il peso bilanciato delle classi (class_weight = balanced) e' quello della pipeline del 2.2c: ogni classe
    pesa allo stesso modo nella loss, come nella macro-F1 [S8 p. 20].

Esecuzione, dalla cartella codice/ (dopo s2_2c_tuning.py):
    python s2_3a_subreddit.py
"""

from __future__ import annotations

import logging
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from comune import FASCE, FILE_COMMENTI, RISULTATI, SEED, carica_split

TAB, FIG = RISULTATI / "tables", RISULTATI / "figures"
FILE_MODELLO = RISULTATI / "pipeline_migliore.joblib"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
N_JOBS = -1
# Scala logaritmica [S8 p. 38]. Le frazioni non standardizzate rendono meglio con C ~ 1, quelle standardizzate
# con C molto piccoli: la griglia copre entrambe e il migliore non deve cadere sul bordo.
GRIGLIA_C_SUBREDDIT = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
GRIGLIA_C_ENTRAMBI = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
GRIGLIA_MIN_UTENTI = [2, 5, 10]
GRIGLIA_SCALA = ["passthrough", StandardScaler(with_mean=False)]    # solo per "solo subreddit"
# Con le frazioni standardizzate e C grande liblinear converge lentamente (fino a ~35.000 iterazioni):
# il limite e' alto perche' ogni configurazione confrontata deve essere arrivata a convergenza [S9 p. 22]
MAX_ITER = 100_000
N_PESI = 15               # subreddit con peso maggiore verso ciascuna classe
log = logging.getLogger("ML4N")


class FrazioniSubreddit(BaseEstimator, TransformerMixin):
    """Da 'documento dei subreddit' (un nome di subreddit per commento, separati da spazi) a frazioni.

    fit: tiene i subreddit con almeno `min_utenti` utenti tra quelli ricevuti (il training).
    transform: frazione dei commenti dell'utente in ciascun subreddit tenuto. Il denominatore e' il numero
    totale di commenti dell'utente, quindi i subreddit esclusi riducono la somma della riga sotto 1.
    """

    def __init__(self, min_utenti: int = 5):
        self.min_utenti = min_utenti

    def fit(self, X, y=None):
        # analyzer=str.split: i nomi dei subreddit non contengono spazi e restano invariati (maiuscole comprese)
        self.vett_ = CountVectorizer(analyzer=str.split, min_df=self.min_utenti).fit(X)
        return self

    def transform(self, X):
        conteggi = self.vett_.transform(X).astype(np.float64)
        totali = np.array([max(len(d.split()), 1) for d in X], dtype=np.float64)
        return sparse.csr_matrix(sparse.diags(1.0 / totali) @ conteggi)

    def get_feature_names_out(self, input_features=None):
        return self.vett_.get_feature_names_out()


def aggiungi_subreddit(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per ogni utente, la colonna `subreddit_doc`: i subreddit dei suoi commenti, uno per commento."""
    commenti = pd.read_csv(FILE_COMMENTI, keep_default_na=False, usecols=["author", "subreddit"])
    doc = commenti.groupby("author")["subreddit"].apply(" ".join).rename("subreddit_doc")
    n = commenti.groupby("author").size().rename("n_commenti_sub")
    for df in (train, test):
        df["subreddit_doc"] = df["author"].map(doc).fillna("")
        diversi = (df["author"].map(n) != df["n_commenti"]).sum()
        if diversi:
            log.warning("%d utenti con numero di commenti diverso tra utenti.csv e supervised_preproc.csv", diversi)
    return train, test


def pipeline_testo(base: Pipeline) -> Pipeline:
    """La pipeline del 2.2c, non addestrata, con gli stessi iperparametri."""
    return clone(base)


def pipeline_subreddit() -> Pipeline:
    return Pipeline([("subreddit", FrazioniSubreddit()), ("scala", "passthrough"),
                     ("modello", LinearSVC(class_weight="balanced", max_iter=MAX_ITER, random_state=SEED))])


def pipeline_entrambi(base: Pipeline) -> Pipeline:
    testo = Pipeline([(nome, clone(passo)) for nome, passo in base.steps[:-1]])     # BoW + TF-IDF del 2.2c
    feature = ColumnTransformer([("testo", testo, "doc"), ("subreddit", FrazioniSubreddit(), "subreddit_doc")])
    return Pipeline([("feature", feature), ("scala", StandardScaler(with_mean=False)),
                     ("modello", LinearSVC(class_weight="balanced", max_iter=MAX_ITER, random_state=SEED))])


def valuta(nome: str, pipe: Pipeline, colonne, train, test, cv_media: float, cv_std: float, param: str) -> dict:
    """Addestra sul training completo e valuta UNA volta sul test."""
    Xtr, Xte = train[colonne], test[colonne]
    t0 = time.perf_counter()
    pipe.fit(Xtr, train["gender"])
    tempo = time.perf_counter() - t0
    pred = pipe.predict(Xte)
    test[f"pred_{nome}"] = pred
    modello = pipe.named_steps["modello"]
    iterazioni = int(np.max(modello.n_iter_))
    y = test["gender"]
    return {"feature": nome, "iperparametri": param, "macro-F1 CV": round(cv_media, 4), "dev. std CV": round(cv_std, 4),
            "macro-F1 training": round(f1_score(train["gender"], pipe.predict(Xtr), average="macro"), 4),
            "macro-F1 test": round(f1_score(y, pred, average="macro"), 4),
            "accuracy test": round(accuracy_score(y, pred), 4),
            "F1 femminile test": round(f1_score(y, pred, pos_label=1), 4),
            "tempo training [s]": round(tempo, 2), "iterazioni": iterazioni,
            "converge": "si'" if iterazioni < modello.max_iter else "NO"}


def cerca(nome: str, pipe: Pipeline, griglia: dict, X, y) -> GridSearchCV:
    gs = GridSearchCV(pipe, griglia, scoring="f1_macro", cv=CV, n_jobs=N_JOBS, return_train_score=True)
    gs.fit(X, y)
    i = gs.best_index_
    log.info("%s: migliori %s, macro-F1 CV %.4f ± %.4f", nome, gs.best_params_,
             gs.cv_results_["mean_test_score"][i], gs.cv_results_["std_test_score"][i])
    tab = pd.DataFrame(gs.cv_results_["params"])
    tab["macro-F1 validazione"] = gs.cv_results_["mean_test_score"].round(4)
    tab["dev. std validazione"] = gs.cv_results_["std_test_score"].round(4)
    tab["macro-F1 training"] = gs.cv_results_["mean_train_score"].round(4)
    tab = tab.astype(str)
    tab.to_csv(TAB / f"subreddit_cv_{nome}.csv", index=False)
    # Se il migliore e' sul bordo della griglia di C, la griglia va allargata
    c = gs.best_params_["modello__C"]
    if c in (min(griglia["modello__C"]), max(griglia["modello__C"])):
        log.warning("%s: il C migliore (%g) e' sul bordo della griglia: allargarla", nome, c)
    return gs


def pesi_subreddit(pipe: Pipeline) -> pd.DataFrame:
    """Subreddit con il peso maggiore verso 'femminile' (positivo) e verso 'maschile' (negativo)."""
    nomi = pipe.named_steps["subreddit"].get_feature_names_out()
    w = pipe.named_steps["modello"].coef_.ravel()
    ordine = np.argsort(w)
    righe = [{"verso": "femminile", "subreddit": nomi[i], "peso": round(float(w[i]), 3)} for i in ordine[::-1][:N_PESI]]
    righe += [{"verso": "maschile", "subreddit": nomi[i], "peso": round(float(w[i]), 3)} for i in ordine[:N_PESI]]
    return pd.DataFrame(righe)


def per_fascia(test: pd.DataFrame, nomi: list[str]) -> pd.DataFrame:
    righe = []
    for f in FASCE + ["tutti"]:
        g = test if f == "tutti" else test[test["fascia_commenti"].astype(str) == f]
        riga = {"fascia": f, "utenti": len(g), "femminile": int(g["gender"].sum())}
        for n in nomi:
            riga[f"macro-F1 {n}"] = round(f1_score(g["gender"], g[f"pred_{n}"], average="macro"), 3)
        righe.append(riga)
    return pd.DataFrame(righe)


def grafico(fasce: pd.DataFrame, nomi: list[str]) -> None:
    t = fasce
    x = np.arange(len(t))
    larg = 0.8 / len(nomi)
    colori = ["#2f6fd6", "#eb6834", "#1baa7a"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, (n, c) in enumerate(zip(nomi, colori)):
        v = t[f"macro-F1 {n}"].to_numpy()
        ax.bar(x + (k - (len(nomi) - 1) / 2) * larg, v, width=larg * 0.92, color=c, label=n)
        for xi, vi in zip(x, v):
            ax.annotate(f"{vi:.2f}", (xi + (k - (len(nomi) - 1) / 2) * larg, vi), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xticks(x, [f"{f}\n(n={u}, F={nf})" for f, u, nf in zip(t["fascia"], t["utenti"], t["femminile"])])
    ax.set_ylim(0.3, 1.0); ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Commenti per utente (utenti di test)"); ax.set_ylabel("macro-F1 sul test")
    ax.set_title("Testo, subreddit o entrambi? (Linear SVM, class_weight = balanced)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "subreddit_confronto.png", dpi=150, bbox_inches="tight")
    plt.close()


def md(df: pd.DataFrame) -> str:
    """Tabella Markdown senza dipendenze esterne (to_markdown richiederebbe il pacchetto tabulate)."""
    righe = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    righe += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(righe)


def main() -> int:
    for p in (TAB, FIG):
        p.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(RISULTATI / "esecuzione_s2_3a.log", encoding="utf-8")])
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)

    if not FILE_MODELLO.is_file():
        raise FileNotFoundError(f"Manca {FILE_MODELLO}: eseguire prima s2_2c_tuning.py")
    base = joblib.load(FILE_MODELLO)          # serve solo per copiare gli iperparametri (clone = non addestrata)
    log.info("Pipeline del 2.2c: %s", [(n, p) for n, p in base.steps])

    train, test = carica_split()
    train, test = aggiungi_subreddit(train, test)
    y = train["gender"]
    n_sub = train["subreddit_doc"].str.split().explode().nunique()
    log.info("Utenti: %d training, %d test; subreddit distinti nel training: %d", len(train), len(test), n_sub)

    # A. solo testo: stessi iperparametri del 2.2c, CV ripetuta con gli stessi fold per avere media e dev. std
    cv_a = cross_validate(pipeline_testo(base), train["doc"], y, scoring="f1_macro", cv=CV, n_jobs=N_JOBS)
    righe = [valuta("solo testo", pipeline_testo(base), "doc", train, test,
                    cv_a["test_score"].mean(), cv_a["test_score"].std(), "come nel 2.2c")]

    # B. solo subreddit
    gs_b = cerca("solo_subreddit", pipeline_subreddit(),
                 {"subreddit__min_utenti": GRIGLIA_MIN_UTENTI, "scala": GRIGLIA_SCALA, "modello__C": GRIGLIA_C_SUBREDDIT},
                 train["subreddit_doc"], y)
    i = gs_b.best_index_
    pipe_b = gs_b.best_estimator_
    righe.append(valuta("solo subreddit", clone(pipe_b), "subreddit_doc", train, test,
                        gs_b.cv_results_["mean_test_score"][i], gs_b.cv_results_["std_test_score"][i],
                        f"min_utenti = {gs_b.best_params_['subreddit__min_utenti']}, C = {gs_b.best_params_['modello__C']:g}, "
                        f"{'standardizzate' if gs_b.best_params_['scala'] != 'passthrough' else 'non standardizzate'}"))

    # C. testo + subreddit
    gs_c = cerca("testo_e_subreddit", pipeline_entrambi(base),
                 {"feature__subreddit__min_utenti": GRIGLIA_MIN_UTENTI, "modello__C": GRIGLIA_C_ENTRAMBI},
                 train[["doc", "subreddit_doc"]], y)
    i = gs_c.best_index_
    righe.append(valuta("testo + subreddit", clone(gs_c.best_estimator_), ["doc", "subreddit_doc"], train, test,
                        gs_c.cv_results_["mean_test_score"][i], gs_c.cv_results_["std_test_score"][i],
                        f"min_utenti = {gs_c.best_params_['feature__subreddit__min_utenti']}, "
                        f"C = {gs_c.best_params_['modello__C']:g}"))

    confronto = pd.DataFrame(righe)
    confronto.to_csv(TAB / "subreddit_confronto.csv", index=False)

    # Copertura: utenti di test senza nessun subreddit tra quelli tenuti (vettore nullo)
    sub_b = pipe_b.named_steps["subreddit"]
    vuoti = int((sub_b.transform(test["subreddit_doc"]).getnnz(axis=1) == 0).sum())
    n_colonne = len(sub_b.get_feature_names_out())

    pesi = pesi_subreddit(pipe_b)
    pesi.to_csv(TAB / "subreddit_pesi.csv", index=False)
    nomi = ["solo testo", "solo subreddit", "testo + subreddit"]
    fasce = per_fascia(test, nomi)
    fasce.to_csv(TAB / "subreddit_per_fascia.csv", index=False)
    grafico(fasce, nomi)

    log.info("\n%s", confronto.to_string(index=False))
    log.info("\n%s", fasce.to_string(index=False))
    log.info("\n%s", pesi.to_string(index=False))

    testo = f"""# ML4N Project 7 — Sezione 2.3a: feature dei subreddit

Feature: per ogni utente, la frazione dei suoi commenti scritta in ciascun subreddit. Subreddit tenuti: quelli
con almeno `min_utenti` utenti di training (scelto in CV). Subreddit distinti nel training: {n_sub:,}; colonne
del modello "solo subreddit": {n_colonne:,}. Utenti di test senza nessun subreddit tenuto: {vuoti}.

Modello: Linear SVM con class_weight = balanced, come la pipeline scelta nel punto 2.2c. Per "solo subreddit"
e "testo + subreddit" C e min_utenti (e per "solo subreddit" anche la standardizzazione) sono scelti con CV a 5
fold sul training (macro-F1, configurazione con la validazione migliore). In "testo + subreddit" la parte di testo
usa gli iperparametri del 2.2c e tutte le colonne sono standardizzate (StandardScaler, with_mean=False), perche'
TF-IDF e frazioni hanno scale diverse. Il test e' usato una sola volta per pipeline.

## Confronto
{md(confronto)}

## Per numero di commenti (test)
{md(fasce)}

## Subreddit con il peso maggiore (modello "solo subreddit")
{md(pesi)}

Grafico: `figures/subreddit_confronto.png`. Griglie complete: `tables/subreddit_cv_*.csv`.
"""
    (RISULTATI / "report_s2_3a.md").write_text(testo, encoding="utf-8")
    log.info("Report: %s", RISULTATI / "report_s2_3a.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
