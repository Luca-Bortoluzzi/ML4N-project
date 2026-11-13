#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# # ML4N Project 7 — Fase 2c: PCA (procedura del lab_4)
#
# Terzo dei tre script della Fase 2 (eseguire prima **fase2a_bow** e **fase2b_tfidf**).
#
# | Script | Traccia | Contenuto |
# |---|---|---|
# | fase2a_bow | 1.3a, 1.3b, 1.4a | aggregazione per autore, BoW, soglie sul vocabolario |
# | fase2b_tfidf | 1.3c, 1.3d | TF-IDF (formula della slide 84) e confronto con la BoW |
# | **fase2c_pca** (questo) | 1.4b | PCA come nel lab_4, PCA senza standardizzazione, TruncatedSVD |
#
# Fonti: *4_DimensionalityReduction* (criteri per il numero di componenti, dipendenza dalla scala) e
# dimostrazione della PCA (le componenti sono gli autovettori della matrice di covarianza, la varianza lungo
# ciascuna è il suo autovalore); **lab_4**, punti 2.1–2.5, replicati uno per uno:
#
# | lab_4 | qui |
# |---|---|
# | 2.1 `StandardScaler` | §1 |
# | 2.3 PCA, varianza cumulativa, scelta delle componenti | §2 |
# | 2.3 loading score | §3 |
# | 2.4 correlazione dopo la PCA | §4 |
# | 2.5 PCA senza standardizzazione | §6 |
#
# **Input**: `tfidf.npz`, `bow.npz`, `vocabolario.csv`, `utenti.csv`.

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # grafici salvati su file (terminale del cluster)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

CARTELLA = Path(__file__).resolve().parent   # lo script sta in cluster/
OUT = CARTELLA / "risultati_fase2" / "vettorizzazione"
FIG, TAB = OUT / "figures", OUT / "tables"
for f in (OUT / "tfidf.npz", OUT / "bow.npz", OUT / "vocabolario.csv", OUT / "utenti.csv"):
    if not f.is_file():
        raise FileNotFoundError(f"Manca {f}: esegui prima fase2a_bow.py e fase2b_tfidf.py")

RANDOM_STATE_PCA = 15      # stesso valore del lab_4
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 250)


def display(*oggetti) -> None:
    """Stampa tabelle e valori a terminale (equivalente testuale del display dei script)."""
    for o in oggetti:
        print(o.to_string() if isinstance(o, (pd.DataFrame, pd.Series)) else o)
        print()



def salva_figura(nome: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / f"{nome}.png", dpi=150, bbox_inches="tight")
    plt.close()


def salva_tabella(df: pd.DataFrame, nome: str, index: bool = False) -> None:
    df.to_csv(TAB / f"{nome}.csv", index=index, encoding="utf-8")


X_tfidf = sparse.load_npz(OUT / "tfidf.npz").tocsr()
X_bow = sparse.load_npz(OUT / "bow.npz").tocsr()
vocab = pd.read_csv(OUT / "vocabolario.csv", keep_default_na=False)["parola"].to_numpy()
utenti = pd.read_csv(OUT / "utenti.csv", keep_default_na=False, usecols=["author", "gender", "n_commenti", "n_token"])
print(f"TF-IDF: {X_tfidf.shape[0]:,} utenti × {X_tfidf.shape[1]:,} parole")

# ## 1. Standardizzazione (lab_4, 2.1)
# Le slide: *"PCA should be applied on data with approximately the same scale"*. Ogni parola (colonna) viene
# portata a media 0 e varianza 1 con `StandardScaler`.
#
# La PCA deve **centrare** i dati: una matrice sparsa diventa **densa**. È il problema citato dalla traccia:
# qui la matrice densa (5.000 × |vocabolario|) entra nella RAM del cluster; per i commenti dell'unsupervised
# no, e servirà la TruncatedSVD (§7).

# float32 dimezza la memoria rispetto a float64 senza perdita di precisione rilevante per la PCA
X_dense = X_tfidf.toarray().astype(np.float32)
mb_sparsa = (X_tfidf.data.nbytes + X_tfidf.indices.nbytes + X_tfidf.indptr.nbytes) / 1e6
print(f"Matrice densa: {X_dense.nbytes / 1e9:.2f} GB (sparsa: {mb_sparsa:.0f} MB)")
X_std = StandardScaler().fit_transform(X_dense)
del X_dense

# ## 2. Varianza spiegata e numero di componenti (lab_4, 2.3)
# Criteri delle slide:
# - **gomito** dello scree plot;
# - **soglia** sulla varianza cumulativa (le slide citano il 66%);
# - **Kaiser**: componenti con autovalore > 1 (su dati standardizzati ogni feature ha varianza 1, quindi si
#   tengono le componenti che spiegano più varianza di una singola parola).

# Calcolo pesante (SVD completa di una matrice 5.000 × |vocabolario|): sul cluster qualche minuto.
t0 = time.perf_counter()
pca = PCA(random_state=RANDOM_STATE_PCA)
pca.fit(X_std)
print(f"PCA: {time.perf_counter() - t0:.1f} s — componenti calcolate: {pca.n_components_:,}")

perc_cumul_exp_var = np.cumsum(pca.explained_variance_ratio_) * 100
SOGLIA_VARIANZA = 66
n_soglia = int(np.searchsorted(perc_cumul_exp_var, SOGLIA_VARIANZA) + 1)
n_kaiser = int((pca.explained_variance_ > 1).sum())
criteri = pd.DataFrame({
    "criterio": [f"varianza cumulativa ≥ {SOGLIA_VARIANZA}%", "Kaiser (autovalore > 1)",
                 "varianza della PC1", "varianza delle prime 10 PC", "varianza delle prime 100 PC"],
    "valore": [f"{n_soglia} componenti",
               f"{n_kaiser} componenti ({perc_cumul_exp_var[n_kaiser - 1]:.1f}% della varianza)",
               f"{perc_cumul_exp_var[0]:.2f}%", f"{perc_cumul_exp_var[9]:.2f}%", f"{perc_cumul_exp_var[99]:.2f}%"]})
salva_tabella(criteri, "pca_criteri_scelta_componenti")
display(criteri)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(np.arange(1, len(perc_cumul_exp_var) + 1), perc_cumul_exp_var)
axes[0].axhline(SOGLIA_VARIANZA, color="tab:red", ls="--", lw=1, label=f"{SOGLIA_VARIANZA}%")
axes[0].axvline(n_soglia, color="tab:red", ls=":", lw=1, label=f"{n_soglia} PC (soglia)")
axes[0].axvline(n_kaiser, color="tab:green", ls=":", lw=1, label=f"{n_kaiser} PC (Kaiser)")
axes[0].set_xlabel("# Principal Components (PCs)"); axes[0].set_ylabel("Cumulative explained variance [%]")
axes[0].set_title(rf"{n_soglia} PCs explain {perc_cumul_exp_var[n_soglia - 1]:.2f}% of $\sigma^2$")
axes[0].legend()
n_scree = 100
axes[1].plot(np.arange(1, n_scree + 1), pca.explained_variance_[:n_scree], marker=".")
axes[1].axhline(1, color="tab:green", ls="--", lw=1, label="autovalore = 1 (Kaiser)")
axes[1].set_yscale("log")
axes[1].set_xlabel("# Principal Component"); axes[1].set_ylabel("Autovalore (scala log)")
axes[1].set_title(f"Scree plot — prime {n_scree} componenti (gomito)"); axes[1].legend()
salva_figura("pca_varianza_cumulativa_scree")

# **Numero di componenti scelto.** Modifica `N_COMPONENTI` dopo aver guardato gomito, soglia e Kaiser.

N_COMPONENTI = n_soglia
X_pca = pca.transform(X_std)[:, :N_COMPONENTI]
print(f"Componenti scelte: {N_COMPONENTI} ({perc_cumul_exp_var[N_COMPONENTI - 1]:.1f}% della varianza) "
      f"al posto di {X_std.shape[1]:,} parole")

# ## 3. Loading score e interpretazione fisica delle componenti (lab_4, 2.3)
# `loadings = pca.components_.T * np.sqrt(pca.explained_variance_)`: per dati standardizzati è la correlazione tra
# ogni parola e la componente. Le parole con loading più alto (positivo) e più basso (negativo) indicano
# **che cosa misura** la componente.
#
# Le prime componenti di una PCA ricalcolata con `n_components=N_COMPONENTI` coincidono con quelle della PCA
# completa, quindi i loading si leggono direttamente da `pca`.

N_PC_LOADING = 10
loadings = pd.DataFrame(pca.components_[:N_PC_LOADING].T * np.sqrt(pca.explained_variance_[:N_PC_LOADING]),
                        columns=[f"PC{i}" for i in range(1, N_PC_LOADING + 1)], index=vocab)
k = 12
tab_loading = pd.DataFrame([{
    "componente": pc,
    "varianza [%]": round(100 * pca.explained_variance_ratio_[i], 2),
    "loading più positivi": ", ".join(f"{w} ({v:.2f})" for w, v in loadings[pc].nlargest(k).items()),
    "loading più negativi": ", ".join(f"{w} ({v:.2f})" for w, v in loadings[pc].nsmallest(k).items()),
} for i, pc in enumerate(loadings.columns)])
salva_tabella(tab_loading, "pca_loading_top_parole")
with pd.option_context("display.max_colwidth", None):
    display(tab_loading)

# Grafico a barre orizzontali come nel lab_4 per PC1 e PC2
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, pc in zip(axes, ["PC1", "PC2"]):
    sel = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index[:20]).sort_values()
    sel.plot.barh(ax=ax, color=np.where(sel > 0, "tab:blue", "tab:orange"))
    ax.set_xlabel("Loading"); ax.set_ylabel("Parola (stem)")
    ax.set_title(f"{pc} — 20 parole con |loading| maggiore")
salva_figura("pca_loading_pc1_pc2")

# Due verifiche numeriche aiutano l'interpretazione:
# - **quanto ogni componente dipende dalla quantità di testo dell'utente** (correlazione di Spearman con il numero
#   di token): se è alta, la componente misura "quanto scrive" l'utente più che "di cosa scrive";
# - **quanto ogni componente è legata al genere** (correlazione di Pearson con l'etichetta 0/1): anticipa quali
#   direzioni saranno utili nella classificazione.

legame = pd.DataFrame({
    "componente": [f"PC{i}" for i in range(1, N_PC_LOADING + 1)],
    "Spearman con n_token": [pd.Series(X_pca[:, i]).corr(utenti["n_token"], method="spearman") for i in range(N_PC_LOADING)],
    "Pearson con gender": [np.corrcoef(X_pca[:, i], utenti["gender"])[0, 1] for i in range(N_PC_LOADING)],
    "loading > 0 [%]": [round(100 * (loadings[pc] > 0).mean(), 1) for pc in loadings.columns],
}).round(3)
salva_tabella(legame, "pca_componenti_vs_lunghezza_e_genere")
display(legame)

# ## 4. Correlazione dopo la PCA (lab_4, 2.4)
# Le componenti principali sono **scorrelate** per costruzione (autovettori ortogonali della matrice di covarianza).

corr_pc = pd.DataFrame(X_pca[:, :10], columns=[f"PC{i}" for i in range(1, 11)]).corr().abs()
plt.figure(figsize=(7, 5.5))
sns.heatmap(corr_pc, cmap="Blues", annot=True, fmt=".2f", vmin=0, vmax=1, cbar_kws={"label": "Correlation"})
plt.title("Correlazione (|Pearson|) tra le prime 10 componenti")
salva_figura("pca_correlazione_componenti")

# ## 5. Visualizzazione su PC1–PC2
# Come l'esempio Iris delle slide (*2_DataExploration*, "Visualization: PCA"): ogni punto è un utente.

def limiti(v: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(v, [0.5, 99.5])
    margine = 0.05 * (hi - lo)
    return lo - margine, hi + margine


pc_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
                      "genere": utenti["gender"].map({0: "maschile (0)", 1: "femminile (1)"})})
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pc_df, x="PC1", y="PC2", hue="genere", s=10, alpha=0.5, linewidth=0)
plt.xlim(limiti(pc_df["PC1"])); plt.ylim(limiti(pc_df["PC2"]))
plt.title("Utenti proiettati su PC1 e PC2 (assi limitati al 0,5–99,5° percentile)")
salva_figura("pca_scatter_pc1_pc2_genere")

# ## 6. PCA senza standardizzazione (lab_4, 2.5)
# Ripetiamo la PCA sulla **BoW grezza** (conteggi non standardizzati). La PCA lavora sulla covarianza, quindi
# le parole con varianza maggiore (le più frequenti) dominano le prime componenti: lo stesso fenomeno "anomalo"
# osservato nel lab_4.

t0 = time.perf_counter()
X_bow_dense = X_bow.toarray().astype(np.float32)
pca_raw = PCA(random_state=RANDOM_STATE_PCA).fit(X_bow_dense)
del X_bow_dense
print(f"PCA sulla BoW grezza: {time.perf_counter() - t0:.1f} s")

cum_raw = np.cumsum(pca_raw.explained_variance_ratio_) * 100
n_soglia_raw = int(np.searchsorted(cum_raw, SOGLIA_VARIANZA) + 1)
loading_raw_pc1 = pd.Series(pca_raw.components_[0] * np.sqrt(pca_raw.explained_variance_[0]), index=vocab)
frequenza_totale = pd.Series(np.asarray(X_bow.sum(axis=0)).ravel(), index=vocab)
rho = loading_raw_pc1.abs().corr(frequenza_totale, method="spearman")
print(f"Senza standardizzazione: {n_soglia_raw} componenti per il {SOGLIA_VARIANZA}% (standardizzata: {n_soglia}); "
      f"PC1 spiega il {cum_raw[0]:.1f}% (standardizzata: {perc_cumul_exp_var[0]:.2f}%)")
print(f"Spearman tra |loading PC1| e frequenza totale della parola: {rho:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].plot(np.arange(1, len(perc_cumul_exp_var) + 1), perc_cumul_exp_var, label="TF-IDF standardizzata")
axes[0].plot(np.arange(1, len(cum_raw) + 1), cum_raw, label="BoW grezza (non standardizzata)")
axes[0].axhline(SOGLIA_VARIANZA, color="tab:red", ls="--", lw=1)
axes[0].set_xscale("log")
axes[0].set_xlabel("# Principal Components (scala log)"); axes[0].set_ylabel("Cumulative explained variance [%]")
axes[0].set_title("Effetto della standardizzazione sulla PCA"); axes[0].legend()
sel = loading_raw_pc1.reindex(loading_raw_pc1.abs().sort_values(ascending=False).index[:20]).sort_values()
sel.plot.barh(ax=axes[1])
axes[1].set_xscale("symlog")
axes[1].set_xlabel("Loading PC1 (scala symlog)"); axes[1].set_title("PC1 della BoW grezza — 20 parole con |loading| maggiore")
salva_figura("pca_senza_standardizzazione")

# ## 7. Alternativa per matrici sparse: TruncatedSVD
# La SVD (citata nelle slide tra le tecniche lineari non supervisionate) **non centra** i dati: lavora sulla
# matrice sparsa senza renderla densa. Sarà la tecnica da usare sui commenti dell'unsupervised (Sezione 3).

N_SVD = min(500, X_tfidf.shape[1] - 1)
t0 = time.perf_counter()
svd = TruncatedSVD(n_components=N_SVD, random_state=RANDOM_STATE_PCA).fit(X_tfidf)
print(f"TruncatedSVD ({N_SVD} componenti) sulla matrice sparsa: {time.perf_counter() - t0:.1f} s")
cum_svd = np.cumsum(svd.explained_variance_ratio_) * 100

plt.figure(figsize=(8, 4.5))
plt.plot(np.arange(1, N_SVD + 1), perc_cumul_exp_var[:N_SVD], label="PCA (TF-IDF standardizzata)")
plt.plot(np.arange(1, N_SVD + 1), cum_svd, label="TruncatedSVD (TF-IDF sparsa, non centrata)")
plt.xlabel("# Componenti"); plt.ylabel("Cumulative explained variance [%]")
plt.title(f"PCA vs TruncatedSVD — prime {N_SVD} componenti"); plt.legend()
salva_figura("pca_vs_svd")

svd_top = pd.DataFrame({f"SVD comp. {i + 1}": vocab[np.argsort(-np.abs(svd.components_[i]))[:12]] for i in range(4)})
salva_tabella(svd_top, "svd_top_parole")
display(svd_top)

# **✍️ Risposta 1.4b (da completare con i numeri sopra).** Punti da toccare:
# - **Funziona?** Computazionalmente sì su questo dataset; non sui commenti singoli dell'unsupervised (densità) →
#   TruncatedSVD. Come compressione: la varianza cresce lentamente, servono molte componenti.
# - **Quante componenti?** Confronta gomito, soglia del 66% e Kaiser e motiva `N_COMPONENTI`.
# - **Interpretazione fisica:** PC1 (loading quasi tutti positivi, forte correlazione con `n_token`) misura
#   *quanto scrive* l'utente; le componenti successive sono *argomenti* riconoscibili dalle parole con loading
#   alto/basso. Oltre le prime decine le componenti mescolano molti argomenti e non sono più interpretabili.
# - **Scala:** senza standardizzazione la PC1 riflette la frequenza delle parole (lab_4, 2.5).

pd.DataFrame([{"N_COMPONENTI": N_COMPONENTI, "varianza_spiegata_%": round(perc_cumul_exp_var[N_COMPONENTI - 1], 2),
               "n_kaiser": n_kaiser, "n_soglia_66": n_soglia, "n_soglia_66_senza_std": n_soglia_raw}]
             ).to_csv(TAB / "pca_scelte.csv", index=False)
print(f"Figure in {FIG}, tabelle in {TAB}")
