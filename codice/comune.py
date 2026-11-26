# -*- coding: utf-8 -*-
"""ML4N Project 7 — Sezione 2: elementi comuni a tutti gli script supervisionati.

Contiene SOLO definizioni (costanti, caricamento dei dati, trasformatori e pipeline): viene importato da
s2_1_dataset.py e dagli script successivi (modelli, tuning, shortcut analysis), cosi' tutti usano
esattamente lo stesso split, la stessa vettorizzazione e gli stessi parametri.

Regola seguita (7_ModelValidationSelection, "Never train on validation… pipeline concept"):
tutto cio' che impara qualcosa dai dati — vocabolario, soglie MIN_DF/MAX_DF, idf, modello — sta dentro
una Pipeline e viene addestrato SOLO sugli utenti di training (o sui fold di training della
cross-validation). Le matrici bow.npz / tfidf.npz della Sezione 1 NON si usano qui: sono state costruite su
tutti i 5.000 utenti e conterrebbero informazione del test set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------
# Percorsi (tutti relativi alla cartella codice/)
# ----------------------------------------------------------------------
CARTELLA = Path(__file__).resolve().parent
FILE_UTENTI = CARTELLA / "risultati_sezione1" / "vettorizzazione" / "utenti.csv"     # da s1_3_bow.py
FILE_COMMENTI = CARTELLA / "risultati_sezione1" / "preprocessing" / "supervised_preproc.csv"           # da s1_2_preprocessing.py
RISULTATI = CARTELLA / "risultati_sezione2"
FILE_SPLIT = RISULTATI / "split_utenti.csv"                                       # da s2_1_dataset.py

# ----------------------------------------------------------------------
# Parametri (scelti e motivati nella Sezione 1)
# ----------------------------------------------------------------------
SEED = 42                 # riproducibilita' (random_state in tutti gli script)
QUOTA_TEST = 0.20         # 80% training, 20% test
MIN_DF = 10               # parola presente in almeno 10 utenti (di training)
MAX_DF = 0.7              # parola presente al massimo nel 70% degli utenti (di training)
ETICHETTE = {0: "maschile", 1: "femminile"}
FASCE = ["1-5", "6-20", "21-100", ">100"]


# ----------------------------------------------------------------------
# Caricamento
# ----------------------------------------------------------------------
def carica_utenti() -> pd.DataFrame:
    """Un utente per riga: author, gender, n_commenti, n_token, fascia_commenti, doc.

    `author` serve solo come chiave (split, analisi per fascia): la traccia vieta di usarlo come feature.
    keep_default_na=False: i documenti vuoti restano stringhe vuote e un autore chiamato "NA" resta tale.
    """
    if not FILE_UTENTI.is_file():
        raise FileNotFoundError(f"Manca {FILE_UTENTI}: eseguire prima s1_3_bow.py")
    utenti = pd.read_csv(FILE_UTENTI, keep_default_na=False)
    utenti["gender"] = utenti["gender"].astype(int)
    utenti["fascia_commenti"] = pd.Categorical(utenti["fascia_commenti"], categories=FASCE, ordered=True)
    return utenti


def carica_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restituisce (train, test) usando lo split salvato da s2_1_dataset.py."""
    if not FILE_SPLIT.is_file():
        raise FileNotFoundError(f"Manca {FILE_SPLIT}: eseguire prima s2_1_dataset.py")
    utenti = carica_utenti()
    split = pd.read_csv(FILE_SPLIT, keep_default_na=False, usecols=["author", "insieme"])
    utenti = utenti.merge(split, on="author", how="inner", validate="one_to_one")
    train = utenti[utenti["insieme"] == "train"].reset_index(drop=True)
    test = utenti[utenti["insieme"] == "test"].reset_index(drop=True)
    return train, test


# ----------------------------------------------------------------------
# TF-IDF con la formula della slide 84 (3_DataPreProcessing)
# ----------------------------------------------------------------------
class TfidfSlide(BaseEstimator, TransformerMixin):
    """tf-idf(t, d) = freq(t, d) * log(m / freq(t, D)), applicata a una matrice BoW.

    fit: calcola l'idf SOLO sui documenti ricevuti (gli utenti di training).
    transform: moltiplica ogni colonna della BoW per il suo idf. Nessuna normalizzazione delle righe,
    a differenza di TfidfVectorizer di sklearn (che usa anche un idf diverso).
    """

    def fit(self, X, y=None):
        m = X.shape[0]
        df_t = np.asarray((X > 0).sum(axis=0)).ravel()
        # Ogni parola del vocabolario compare almeno MIN_DF volte nel training, quindi df_t >= 1:
        # il controllo protegge solo da usi impropri.
        df_t = np.maximum(df_t, 1)
        self.idf_ = np.log(m / df_t)
        return self

    def transform(self, X):
        return sparse.csr_matrix(sparse.csr_matrix(X).multiply(self.idf_))


def crea_vettorizzatore(min_df=MIN_DF, max_df=MAX_DF) -> CountVectorizer:
    """BoW sui documenti gia' preprocessati (token separati da spazi)."""
    return CountVectorizer(analyzer=str.split, min_df=min_df, max_df=max_df)


def crea_pipeline(rappresentazione: str, modello) -> Pipeline:
    """Pipeline completa: BoW (+ TF-IDF della slide) + modello.

    Le colonne NON vengono standardizzate: con la formula della slide la TF-IDF e' la BoW con ogni colonna
    moltiplicata per una costante (l'idf), e la standardizzazione renderebbe le due rappresentazioni
    identiche (verificato nella Sezione 1.4, s1_4_pca.py §6.2).
    """
    if rappresentazione == "bow":
        passi = [("bow", crea_vettorizzatore())]
    elif rappresentazione == "tfidf":
        passi = [("bow", crea_vettorizzatore()), ("tfidf", TfidfSlide())]
    else:
        raise ValueError(f"Rappresentazione sconosciuta: {rappresentazione!r} (usare 'bow' o 'tfidf')")
    return Pipeline(passi + [("modello", modello)])
