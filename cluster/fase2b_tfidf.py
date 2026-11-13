#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# # ML4N Project 7 — Fase 2b: TF-IDF e confronto con la BoW
#
# Secondo dei tre script della Fase 2 (eseguire prima **fase2a_bow**).
#
# | Script | Traccia | Contenuto |
# |---|---|---|
# | fase2a_bow | 1.3a, 1.3b, 1.4a | aggregazione per autore, BoW, soglie sul vocabolario |
# | **fase2b_tfidf** (questo) | 1.3c, 1.3d | TF-IDF (formula della slide 84) e confronto con la BoW |
# | fase2c_pca | 1.4b | PCA come nel lab_4, PCA senza standardizzazione, TruncatedSVD |
#
# **Fonte**: *3_DataPreProcessing*, slide 83–84. La TF-IDF si calcola con **esattamente la formula della slide**:
#
# $$\text{tf-idf}(t, d) = \text{freq}(t, d)\cdot \log\frac{m}{\text{freq}(t, D)}$$
#
# con $m$ = numero di documenti (utenti) e $\text{freq}(t, D)$ = numero di documenti che contengono $t$.
# `TfidfVectorizer` di sklearn usa una variante diversa (idf + 1, smoothing, normalizzazione L2 delle righe),
# quindi la TF-IDF si ottiene dalla BoW del script precedente, sullo stesso vocabolario ridotto.
#
# **Input**: `bow.npz`, `vocabolario.csv`, `utenti.csv` (da fase2a). **Output**: `tfidf.npz`.

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse

CARTELLA = Path(__file__).resolve().parent   # lo script sta in cluster/
OUT = CARTELLA / "risultati_fase2" / "vettorizzazione"
TAB = OUT / "tables"
for f in (OUT / "bow.npz", OUT / "vocabolario.csv", OUT / "utenti.csv"):
    if not f.is_file():
        raise FileNotFoundError(f"Manca {f}: esegui prima fase2a_bow.py")

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 250)


def display(*oggetti) -> None:
    """Stampa tabelle e valori a terminale (equivalente testuale del display dei script)."""
    for o in oggetti:
        print(o.to_string() if isinstance(o, (pd.DataFrame, pd.Series)) else o)
        print()



def salva_tabella(df: pd.DataFrame, nome: str, index: bool = False) -> None:
    df.to_csv(TAB / f"{nome}.csv", index=index, encoding="utf-8")


X_bow = sparse.load_npz(OUT / "bow.npz").tocsr()
vocab_info = pd.read_csv(OUT / "vocabolario.csv", keep_default_na=False)
vocab = vocab_info["parola"].to_numpy()
utenti = pd.read_csv(OUT / "utenti.csv", keep_default_na=False, usecols=["author", "gender", "n_commenti", "n_token"])
assert X_bow.shape == (len(utenti), len(vocab))
print(f"BoW caricata: {X_bow.shape[0]:,} utenti × {X_bow.shape[1]:,} parole")

# ## 1. Calcolo della TF-IDF (traccia 1.3c)

def tfidf_slide(X: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    """TF-IDF con la formula della slide 84: freq(t,d) * log(m / freq(t,D))."""
    m = X.shape[0]
    df_t = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log(m / df_t)
    return sparse.csr_matrix(X.multiply(idf)), idf


X_tfidf, idf = tfidf_slide(X_bow)
vocab_info["idf"] = idf

print(f"TF-IDF: {X_tfidf.shape[0]:,} × {X_tfidf.shape[1]:,}, celle non nulle {X_tfidf.nnz:,} (BoW: {X_bow.nnz:,})")
print(f"idf: min {idf.min():.2f} (parola più diffusa), max {idf.max():.2f} (parola più rara)")
display(vocab_info.sort_values("idf").iloc[np.r_[0:5, -5:0]][["parola", "utenti", "idf"]])

# ## 2. Differenze tra BoW e TF-IDF (traccia 1.3d)
#
# La slide 83 avverte che le parole più frequenti *"are not always representative"*; la TF-IDF (slide 84) premia
# i termini frequenti in un documento ma rari nella collezione. Lo verifichiamo in tre modi.
#
# ### 2.1 Parole più pesanti per alcuni utenti di esempio

def top_parole(X: sparse.csr_matrix, riga: int, k: int = 10) -> str:
    r = X.getrow(riga)
    ordine = np.argsort(-r.data)[:k]
    return ", ".join(f"{vocab[r.indices[i]]} ({r.data[i]:.1f})" for i in ordine)


def utente_vicino(mask: pd.Series, valore: float) -> int:
    return int((utenti.loc[mask, "n_commenti"] - valore).abs().idxmin())


mediana = utenti["n_commenti"].median()
esempi_idx = {
    "maschile, n. commenti mediano": utente_vicino(utenti["gender"] == 0, mediana),
    "femminile, n. commenti mediano": utente_vicino(utenti["gender"] == 1, mediana),
    "utente molto attivo (p99)": utente_vicino(utenti["n_commenti"] > 0, utenti["n_commenti"].quantile(.99)),
}
confronto_utenti = pd.DataFrame([
    {"utente": d, "commenti": utenti.loc[i, "n_commenti"],
     "top-10 BoW (conteggio)": top_parole(X_bow, i), "top-10 TF-IDF (peso)": top_parole(X_tfidf, i)}
    for d, i in esempi_idx.items()])
salva_tabella(confronto_utenti, "bow_vs_tfidf_utenti_esempio")
with pd.option_context("display.max_colwidth", None):
    display(confronto_utenti)

# ### 2.2 Parole più pesanti sull'intera collezione

k = 20
tot_bow = np.asarray(X_bow.sum(axis=0)).ravel()
tot_tfidf = np.asarray(X_tfidf.sum(axis=0)).ravel()
top_globali = pd.DataFrame({"top BoW (somma conteggi)": vocab[np.argsort(-tot_bow)[:k]],
                            "top TF-IDF (somma pesi)": vocab[np.argsort(-tot_tfidf)[:k]]})
salva_tabella(top_globali, "bow_vs_tfidf_top_globali")
rho = pd.Series(tot_bow).corr(pd.Series(idf), method="spearman")
print(f"Correlazione di Spearman tra frequenza totale e idf delle parole: {rho:.2f} "
      "(negativa: più una parola è usata, meno pesa nella TF-IDF)")
display(top_globali)

# ### 2.3 Effetto della lunghezza dei documenti
# Con la formula della slide nessuna delle due rappresentazioni normalizza le righe: un utente che scrive molto
# ha un vettore "più grande" in entrambe.

massa = pd.DataFrame({
    "n_token": utenti["n_token"],
    "somma riga BoW": np.asarray(X_bow.sum(axis=1)).ravel(),
    "norma L2 riga TF-IDF": np.sqrt(np.asarray(X_tfidf.multiply(X_tfidf).sum(axis=1)).ravel()),
})
print("Correlazione di Spearman con il numero di token dell'utente:")
display(massa.corr(method="spearman")["n_token"].drop("n_token").round(3))

# **✍️ Risposta 1.3d (da completare con i numeri sopra).** Punti da toccare:
# - **BoW**: pesa ogni parola con la sua frequenza; in testa ci sono parole generiche usate da molti utenti.
# - **TF-IDF**: moltiplica la frequenza per $\log(m/\text{freq}(t,D))$; le parole diffuse vengono penalizzate,
#   quelle specifiche di un utente (argomenti, hobby, squadre, giochi) emergono.
#   Una parola presente in tutti i documenti avrebbe peso nullo.
# - **Sparsità**: identica (stesse celle non nulle): cambia il peso, non quali parole sono presenti.
# - **Lunghezza**: con la formula della slide entrambe crescono con il numero di token dell'utente;
#   è un effetto da ricordare per la PCA (fase2c) e per la classificazione.

sparse.save_npz(OUT / "tfidf.npz", X_tfidf)
vocab_info.to_csv(OUT / "vocabolario.csv", index=False, encoding="utf-8")   # ora con la colonna idf
print(f"Salvati in {OUT}: tfidf.npz, vocabolario.csv (con idf)")
