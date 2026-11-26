#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML4N Project 7 — Sezione 2.1: preparazione del dataset per il training supervisionato.

Operazioni (scaletta, Fase 3, punti 1-5):
  1. carica il dataset per utente (utenti.csv della Sezione 1) e controlla che sia integro;
  2. associa le etichette (gia' presenti) e verifica che `author` non sia usato come feature;
  3. divide gli utenti in training (80%) e test (20%) con split STRATIFICATO per genere
     (7_ModelValidationSelection, "class-ratio preserving splitting") e salva lo split;
  4. calcola la baseline che predice sempre la classe maggioritaria (8_Metrics);
  5. costruisce le pipeline BoW / TF-IDF addestrandole SOLO sul training, e misura cosa cambia rispetto a
     un vocabolario costruito su tutti gli utenti (leakage evitato), le parole del test fuori vocabolario e
     gli utenti con vettore vuoto.

Esecuzione, dalla cartella codice/:
    python s2_1_dataset.py
Output in codice/risultati_sezione2/: split_utenti.csv, tables/, figures/, report_s2_1.md
"""

from __future__ import annotations

import logging
import sys

import matplotlib
matplotlib.use("Agg")  # grafici su file (terminale del cluster)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from comune import (ETICHETTE, FASCE, FILE_SPLIT, MAX_DF, MIN_DF, QUOTA_TEST, RISULTATI, SEED,
                               carica_utenti, crea_pipeline, crea_vettorizzatore)

TAB, FIG = RISULTATI / "tables", RISULTATI / "figures"
log = logging.getLogger("ML4N")


def salva(df: pd.DataFrame, nome: str, index: bool = False) -> None:
    df.to_csv(TAB / f"{nome}.csv", index=index, encoding="utf-8")


def percentuali(serie: pd.Series) -> str:
    return " / ".join(f"{ETICHETTE[k]} {v:.1%}" for k, v in serie.value_counts(normalize=True).sort_index().items())


# ----------------------------------------------------------------------
# 1-2. Caricamento e controlli
# ----------------------------------------------------------------------
def carica_e_controlla() -> tuple[pd.DataFrame, dict]:
    utenti = carica_utenti()
    controlli = {
        "utenti": len(utenti),
        "autori duplicati": int(utenti["author"].duplicated().sum()),
        "etichette non valide": int((~utenti["gender"].isin([0, 1])).sum()),
        "documenti vuoti": int((utenti["n_token"] == 0).sum()),
    }
    assert controlli["autori duplicati"] == 0 and controlli["etichette non valide"] == 0, controlli
    # La traccia vieta di usare `author` come feature: le pipeline ricevono solo la colonna `doc`.
    FEATURE = "doc"
    assert FEATURE != "author"
    log.info("Utenti: %s — %s — documenti vuoti: %s", f"{len(utenti):,}", percentuali(utenti["gender"]),
             controlli["documenti vuoti"])
    return utenti, controlli


# ----------------------------------------------------------------------
# 3. Split stratificato
# ----------------------------------------------------------------------
def split_stratificato(utenti: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(utenti, test_size=QUOTA_TEST, stratify=utenti["gender"], random_state=SEED)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    assert set(train["author"]).isdisjoint(test["author"]), "Lo stesso utente e' in training e in test"

    split = pd.concat([train.assign(insieme="train"), test.assign(insieme="test")], ignore_index=True)
    split[["author", "gender", "n_commenti", "n_token", "fascia_commenti", "insieme"]].to_csv(
        FILE_SPLIT, index=False, encoding="utf-8")
    log.info("Split salvato in %s: %s training, %s test", FILE_SPLIT, f"{len(train):,}", f"{len(test):,}")
    return train, test


def confronta_insiemi(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verifica che training e test abbiano la stessa composizione (classi, fasce, attivita')."""
    righe = []
    for nome, df in (("training", train), ("test", test)):
        righe.append({
            "insieme": nome, "utenti": len(df),
            "maschile (0)": int((df["gender"] == 0).sum()), "femminile (1)": int((df["gender"] == 1).sum()),
            "quota femminile [%]": round(100 * df["gender"].mean(), 2),
            "commenti (mediana)": float(df["n_commenti"].median()),
            "token (mediana)": float(df["n_token"].median()),
            "documenti vuoti": int((df["n_token"] == 0).sum()),
        })
    composizione = pd.DataFrame(righe)
    salva(composizione, "composizione_train_test")

    fasce = (pd.concat([train.assign(insieme="training"), test.assign(insieme="test")])
             .pivot_table(index="fascia_commenti", columns=["insieme", "gender"], values="author",
                          aggfunc="count", observed=False, fill_value=0))
    fasce.columns = [f"{ins} - {ETICHETTE[g]}" for ins, g in fasce.columns]
    fasce = fasce[[c for c in ["training - maschile", "training - femminile", "test - maschile", "test - femminile"]]]
    fasce["test - totale"] = fasce["test - maschile"] + fasce["test - femminile"]
    salva(fasce, "utenti_per_fascia_train_test", index=True)
    return composizione, fasce


def grafico_composizione(train: pd.DataFrame, test: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    # Quota delle classi nei due insiemi (stratificazione)
    insiemi = {"training": train, "test": test}
    for i, (nome, df) in enumerate(insiemi.items()):
        quote = df["gender"].value_counts(normalize=True).sort_index() * 100
        sinistra = 0
        for g, q in quote.items():
            axes[0].barh(nome, q, left=sinistra, color=["#9aa5b1", "#2f6fd6"][g], label=ETICHETTE[g] if i == 0 else None)
            axes[0].text(sinistra + q / 2, nome, f"{ETICHETTE[g]} {q:.1f}%", ha="center", va="center", color="white")
            sinistra += q
    axes[0].set_xlim(0, 100); axes[0].set_xlabel("Quota degli utenti [%]")
    axes[0].set_title("Stessa proporzione di classi in training e test (split stratificato)")
    # Utenti di test per fascia di commenti (servira' al punto 2.2d)
    conteggi = test.groupby(["fascia_commenti", "gender"], observed=False).size().unstack(fill_value=0).reindex(FASCE)
    x = np.arange(len(FASCE))
    axes[1].bar(x - 0.2, conteggi[0], width=0.4, color="#9aa5b1", label="maschile")
    axes[1].bar(x + 0.2, conteggi[1], width=0.4, color="#2f6fd6", label="femminile")
    for xi, (m, f) in zip(x, conteggi.to_numpy()):
        axes[1].text(xi - 0.2, m + 3, str(m), ha="center", fontsize=9)
        axes[1].text(xi + 0.2, f + 3, str(f), ha="center", fontsize=9)
    axes[1].set_xticks(x, FASCE); axes[1].set_xlabel("Commenti per utente"); axes[1].set_ylabel("Utenti di test")
    axes[1].set_title("Utenti di test per fascia di commenti"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG / "composizione_train_test.png", dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
# 4. Baseline: classe maggioritaria
# ----------------------------------------------------------------------
def metriche(y_vero, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_vero, y_pred),
            "F1 maschile": f1_score(y_vero, y_pred, pos_label=0, zero_division=0),
            "F1 femminile": f1_score(y_vero, y_pred, pos_label=1, zero_division=0),
            "macro-F1": f1_score(y_vero, y_pred, average="macro", zero_division=0)}


def baseline(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    dummy = DummyClassifier(strategy="most_frequent")
    pipe = crea_pipeline("bow", dummy).fit(train["doc"], train["gender"])
    righe = []
    for nome, df in (("training", train), ("test", test)):
        pred = pipe.predict(df["doc"])
        righe.append({"insieme": nome, **{k: round(v, 4) for k, v in metriche(df["gender"], pred).items()}})
        cm = pd.DataFrame(confusion_matrix(df["gender"], pred, labels=[0, 1]),
                          index=["vero maschile", "vero femminile"], columns=["pred. maschile", "pred. femminile"])
        salva(cm, f"baseline_confusion_{nome}", index=True)
        log.info("Baseline — %s:\n%s\n%s", nome, cm.to_string(),
                 classification_report(df["gender"], pred, labels=[0, 1], target_names=list(ETICHETTE.values()),
                                       zero_division=0))
    tab = pd.DataFrame(righe)
    salva(tab, "baseline_metriche")
    return tab


# ----------------------------------------------------------------------
# 5. Vettorizzazione addestrata solo sul training
# ----------------------------------------------------------------------
def vettorizzazione(utenti: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    vec_train = crea_vettorizzatore().fit(train["doc"])
    vocab_train = set(vec_train.vocabulary_)
    vocab_tutti = set(crea_vettorizzatore().fit(utenti["doc"]).vocabulary_)

    X_train, X_test = vec_train.transform(train["doc"]), vec_train.transform(test["doc"])
    # Parole del test che il vocabolario di training non conosce (out of vocabulary)
    token_test = test["doc"].str.split().explode().dropna()
    fuori = ~token_test.isin(vocab_train)
    # parole tolte perche' troppo comuni (MAX_DF): note, ma escluse di proposito
    df_train = train["doc"].str.split().apply(set).explode().value_counts()
    troppo_comuni = set(df_train[df_train > MAX_DF * len(train)].index)
    oov_comuni = 100 * (fuori & token_test.isin(troppo_comuni)).mean()
    oov_rare = 100 * (fuori & ~token_test.isin(troppo_comuni)).mean()

    pipe_tfidf = crea_pipeline("tfidf", DummyClassifier(strategy="most_frequent")).fit(train["doc"], train["gender"])
    idf = pipe_tfidf.named_steps["tfidf"].idf_

    def sparsita(X) -> float:
        return round(100 * (1 - X.nnz / (X.shape[0] * X.shape[1])), 2)

    tab = pd.DataFrame([
        {"misura": "parole nel vocabolario (fit sul training)", "valore": f"{len(vocab_train):,}"},
        {"misura": "parole nel vocabolario se fit su tutti i 5.000 utenti", "valore": f"{len(vocab_tutti):,}"},
        {"misura": "parole in piu' con il fit su tutti (scelte anche grazie al test: leakage evitato)", "valore": f"{len(vocab_tutti - vocab_train):,}"},
        {"misura": "parole del vocabolario di training assenti nel fit su tutti", "valore": f"{len(vocab_train - vocab_tutti):,}"},
        {"misura": f"token del test esclusi perche' troppo comuni ({', '.join(sorted(troppo_comuni))}) [%]", "valore": f"{oov_comuni:.1f}"},
        {"misura": "token del test fuori vocabolario perche' rari o mai visti nel training [%]", "valore": f"{oov_rare:.1f}"},
        {"misura": "matrice di training (utenti x parole)", "valore": f"{X_train.shape[0]:,} x {X_train.shape[1]:,}"},
        {"misura": "matrice di test (utenti x parole)", "valore": f"{X_test.shape[0]:,} x {X_test.shape[1]:,}"},
        {"misura": "sparsita' training / test [%]", "valore": f"{sparsita(X_train)} / {sparsita(X_test)}"},
        {"misura": "utenti con vettore vuoto training / test", "valore":
            f"{int((X_train.sum(axis=1) == 0).sum())} / {int((X_test.sum(axis=1) == 0).sum())}"},
        {"misura": "idf calcolato su (utenti di training)", "valore": f"{len(train):,}"},
        {"misura": "idf min / max", "valore": f"{idf.min():.2f} / {idf.max():.2f}"},
    ])
    salva(tab, "vettorizzazione_solo_training")
    return tab


def report(controlli, composizione, fasce, base, vett) -> None:
    def md(df: pd.DataFrame, index: bool = False) -> str:
        # tabella Markdown senza dipendenze esterne (to_markdown richiederebbe il pacchetto tabulate)
        d = df.reset_index() if index else df
        righe = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
        righe += ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
        return "\n".join(righe)

    testo = f"""# ML4N Project 7 — Sezione 2.1: preparazione del dataset

## Dati
- Utenti: {controlli['utenti']:,}; autori duplicati: {controlli['autori duplicati']}; etichette non valide: {controlli['etichette non valide']}.
- Utenti con documento vuoto dopo il preprocessing: {controlli['documenti vuoti']}.
- Feature usata: solo il testo aggregato per utente (`doc`). `author` è usato solo come chiave, come richiesto dalla traccia.

## Split stratificato ({1 - QUOTA_TEST:.0%} / {QUOTA_TEST:.0%}, random_state = {SEED})
{md(composizione)}

Utenti per fascia di commenti (servirà al punto 2.2d):

{md(fasce, index=True)}

## Baseline: predice sempre la classe maggioritaria
{md(base)}

## Vettorizzazione addestrata solo sul training (MIN_DF = {MIN_DF}, MAX_DF = {MAX_DF})
{md(vett)}
"""
    (RISULTATI / "report_s2_1.md").write_text(testo, encoding="utf-8")


def main() -> int:
    for p in (TAB, FIG):
        p.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(RISULTATI / "esecuzione_s2_1.log", encoding="utf-8")])
    pd.set_option("display.width", 200)

    utenti, controlli = carica_e_controlla()
    train, test = split_stratificato(utenti)
    composizione, fasce = confronta_insiemi(train, test)
    grafico_composizione(train, test)
    base = baseline(train, test)
    vett = vettorizzazione(utenti, train, test)

    log.info("Composizione:\n%s", composizione.to_string(index=False))
    log.info("Utenti per fascia:\n%s", fasce.to_string())
    log.info("Baseline:\n%s", base.to_string(index=False))
    log.info("Vettorizzazione:\n%s", vett.to_string(index=False))
    report(controlli, composizione, fasce, base, vett)
    log.info("FINE. Report: %s", RISULTATI / "report_s2_1.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Sezione 2.1 interrotta da un errore")
        raise
