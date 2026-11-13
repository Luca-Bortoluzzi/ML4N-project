#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# # ML4N Project 7 — Fase 2a: aggregazione per autore e Bag-of-Words
#
# Primo dei tre script della Fase 2 (vanno eseguiti in ordine):
#
# | Script | Traccia | Contenuto |
# |---|---|---|
# | **fase2a_bow** (questo) | 1.3a, 1.3b, 1.4a | aggregazione per autore, BoW, soglie sul vocabolario |
# | fase2b_tfidf | 1.3c, 1.3d | TF-IDF (formula della slide 84) e confronto con la BoW |
# | fase2c_pca | 1.4b | PCA come nel lab_4, PCA senza standardizzazione, TruncatedSVD |
#
# Fonti del corso usate qui:
# - **aggregazione**: *3_DataPreProcessing*, slide 40–41; `groupby().agg()` come nel lab_4 (punto 2.1);
# - **BoW**: *3_DataPreProcessing*, slide 80–83 (schema *simple frequency*);
# - **soglie**: *3_DataPreProcessing*, slide 47–48 (curse of dimensionality, feature irrilevanti/ridondanti);
#   ECDF e percentili (*2_DataExploration*, lab_3). La traccia (1.4a) chiede di usare la BoW per identificarle.
#
# **Input**: `risultati_fase2/supervised_preproc.csv` (da `fase2_clean.py`) e `target_supervised.csv`.
# **Output** in `risultati_fase2/vettorizzazione/`: `utenti.csv`, `bow.npz`, `vocabolario.csv`, figure e tabelle.
#
# > **Data leakage.** Qui tutto è calcolato sui 5.000 utenti perché è analisi esplorativa (Sezione 1).
# > Nella Fase 3 vettorizzazione e soglie andranno rifatte dentro una `Pipeline` addestrata solo sul training set
# > (*7_ModelValidationSelection*: "Never train on validation").

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # grafici salvati su file (terminale del cluster)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

CARTELLA = Path(__file__).resolve().parent   # lo script sta in cluster/
FILE_PREPROC = CARTELLA / "risultati_fase2" / "supervised_preproc.csv"
FILE_TARGET = CARTELLA / "target_supervised.csv"
OUT = CARTELLA / "risultati_fase2" / "vettorizzazione"
FIG, TAB = OUT / "figures", OUT / "tables"
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)
for f in (FILE_PREPROC, FILE_TARGET):
    if not f.is_file():
        raise FileNotFoundError(f"Manca {f}: esegui lo script da cluster/ dopo fase2_clean.py")

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)
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


def descrivi_matrice(nome: str, X: sparse.spmatrix) -> dict:
    m, v = X.shape
    return {"rappresentazione": nome, "documenti": m, "parole (feature)": v,
            "celle non nulle": X.nnz, "sparsità [%]": round(100 * (1 - X.nnz / (m * v)), 4)}

# ## 1. Aggregazione dei commenti per autore (traccia 1.3a)
#
# Per la classificazione il campione è l'**utente**: uniamo tutti i suoi commenti preprocessati in un unico
# documento (*aggregation*: *"combining two or more objects into a single object… more stable data"*).
# I commenti vuoti dopo il preprocessing (`n_token = 0`) contano solo nel numero di commenti.

# keep_default_na=False: i clean_text vuoti restano stringhe vuote (non NaN)
commenti = pd.read_csv(FILE_PREPROC, keep_default_na=False, usecols=["author", "clean_text", "n_token"])
target = pd.read_csv(FILE_TARGET, keep_default_na=False)
target["gender"] = target["gender"].astype(int)
print(f"Commenti caricati: {len(commenti):,} — autori: {commenti['author'].nunique():,}")

utenti = (commenti
          .groupby("author", sort=True)
          .agg(doc=("clean_text", lambda s: " ".join(t for t in s if t)),
               n_commenti=("clean_text", "size"),
               n_token=("n_token", "sum"))
          .reset_index()
          .merge(target, on="author", how="inner", validate="one_to_one"))
assert len(utenti) == target["author"].nunique(), "Alcuni autori del target non hanno commenti"
del commenti

FASCE = [0, 5, 20, 100, np.inf]
utenti["fascia_commenti"] = pd.cut(utenti["n_commenti"], bins=FASCE, labels=["1-5", "6-20", "21-100", ">100"])
print(f"Utenti (documenti): {len(utenti):,} — documenti vuoti: {(utenti['n_token'] == 0).sum()}")

# Fasce di commenti per utente: serviranno al punto **2.2d**.

tab_fasce = pd.crosstab(utenti["fascia_commenti"], utenti["gender"], margins=True, margins_name="totale")
tab_fasce.columns = [{0: "maschile (0)", 1: "femminile (1)"}.get(c, c) for c in tab_fasce.columns]
salva_tabella(tab_fasce, "utenti_per_fascia_commenti", index=True)
display(tab_fasce)
desc_utenti = utenti[["n_commenti", "n_token"]].describe(percentiles=[.25, .5, .75, .9, .99]).T
salva_tabella(desc_utenti, "statistiche_per_utente", index=True)
display(desc_utenti)

# ## 2. Bag-of-Words sul vocabolario completo (traccia 1.3b)
#
# Schema *simple frequency* della slide 83: la cella $(d, t)$ è il numero di volte in cui la parola $t$ compare
# nel documento dell'utente $d$. I documenti sono già tokenizzati e in minuscolo, quindi `CountVectorizer`
# usa semplicemente `analyzer=str.split`.

t0 = time.perf_counter()
bow_completa = CountVectorizer(analyzer=str.split)
X_bow_full = bow_completa.fit_transform(utenti["doc"])
vocab_full = bow_completa.get_feature_names_out()
print(f"BoW completa: {time.perf_counter() - t0:.1f} s")

riepilogo = pd.DataFrame([descrivi_matrice("BoW (vocabolario completo)", X_bow_full)])
display(riepilogo)

# ## 3. Soglie sul vocabolario usando la BoW (traccia 1.4a)
#
# Con ~80.000 parole e 5.000 documenti lo spazio è enorme e sparso (*curse of dimensionality*, slide 47).
# Applichiamo la *feature subset selection* (slide 48) usando la **document frequency** (in quanti utenti compare
# ogni parola), calcolata dalla BoW:
# - parole usate da **pochissimi** utenti: refusi, nomi propri, gergo personale → **irrilevanti**
#   per descrivere gruppi di utenti (soglia `MIN_DF`);
# - parole usate da **quasi tutti**: non distinguono nessuno → **irrilevanti** per separare gli utenti
#   (soglia `MAX_DF`).

m = X_bow_full.shape[0]
df_t = np.asarray((X_bow_full > 0).sum(axis=0)).ravel()     # document frequency di ogni parola
occ_t = np.asarray(X_bow_full.sum(axis=0)).ravel()          # occorrenze totali di ogni parola

percentili = [50, 75, 90, 95, 99, 99.9]
tab_percentili = pd.DataFrame({"percentile": percentili, "document frequency": np.percentile(df_t, percentili)})
salva_tabella(tab_percentili, "document_frequency_percentili")
print(f"Parole presenti in 1 solo utente: {(df_t == 1).sum():,} su {len(df_t):,} ({100 * (df_t == 1).mean():.1f}%) "
      f"— ma solo il {100 * occ_t[df_t == 1].sum() / occ_t.sum():.1f}% delle occorrenze")
display(tab_percentili)

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.ecdfplot(x=df_t, log_scale=True, ax=ax)
ax.set_xlabel("Document frequency (n. di utenti che usano la parola, scala log)")
ax.set_ylabel("Frazione cumulativa delle parole")
ax.set_title("ECDF della document frequency (BoW, vocabolario completo)")
salva_figura("ecdf_document_frequency")

# Effetto delle due soglie: parole tenute e quota di occorrenze conservata
candidati_min = [1, 2, 3, 5, 10, 20, 50, 100]
candidati_max = [1.0, 0.9, 0.7, 0.5, 0.3]
vocab_min = pd.DataFrame({
    "min_df (utenti)": candidati_min,
    "parole tenute": [int((df_t >= k).sum()) for k in candidati_min],
    "occorrenze conservate [%]": [round(100 * occ_t[df_t >= k].sum() / occ_t.sum(), 1) for k in candidati_min]})
vocab_max = pd.DataFrame({
    "max_df (frazione utenti)": candidati_max,
    "parole eliminate": [int((df_t > f * m).sum()) for f in candidati_max],
    "occorrenze eliminate [%]": [round(100 * occ_t[df_t > f * m].sum() / occ_t.sum(), 1) for f in candidati_max]})

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(vocab_min.iloc[:, 0], vocab_min.iloc[:, 1], marker="o")
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_xlabel("min_df (n. minimo di utenti)"); axes[0].set_ylabel("Dimensione del vocabolario")
axes[0].set_title("Parole tenute in funzione di min_df")
axes[1].plot(vocab_max.iloc[:, 0], vocab_max.iloc[:, 1], marker="o")
axes[1].invert_xaxis()
axes[1].set_xlabel("max_df (frazione massima di utenti)"); axes[1].set_ylabel("Parole eliminate")
axes[1].set_title("Parole eliminate in funzione di max_df")
salva_figura("vocabolario_vs_soglie")
salva_tabella(vocab_min, "vocabolario_vs_min_df")
salva_tabella(vocab_max, "parole_eliminate_vs_max_df")
display(vocab_min, vocab_max)

# Le parole più diffuse: candidate a essere eliminate con MAX_DF
piu_diffuse = (pd.DataFrame({"parola": vocab_full, "utenti": df_t, "quota utenti [%]": (100 * df_t / m).round(1)})
               .sort_values("utenti", ascending=False).head(30))
salva_tabella(piu_diffuse, "parole_piu_diffuse")
display(piu_diffuse)

# **Scelta delle soglie.** Modifica i due valori dopo aver guardato ECDF, grafici e tabelle, e motiva la scelta
# nella relazione con i numeri delle tabelle (parole tenute, occorrenze conservate, parole eliminate da `MAX_DF`).

MIN_DF = 10      # n. minimo di utenti in cui deve comparire una parola
MAX_DF = 0.5     # frazione massima di utenti (sopra questa quota la parola viene eliminata)

bow = CountVectorizer(analyzer=str.split, min_df=MIN_DF, max_df=MAX_DF)
X_bow = bow.fit_transform(utenti["doc"]).tocsr()
vocab = bow.get_feature_names_out()

eliminate_rare = int((df_t < MIN_DF).sum())
eliminate_comuni = sorted(vocab_full[df_t > MAX_DF * m])
print(f"Vocabolario: {len(vocab_full):,} → {len(vocab):,} parole "
      f"({100 * len(vocab) / len(vocab_full):.1f}% del vocabolario, "
      f"{100 * X_bow.sum() / X_bow_full.sum():.1f}% delle occorrenze)")
print(f"Eliminate perché rare (< {MIN_DF} utenti): {eliminate_rare:,}")
print(f"Eliminate perché troppo comuni (> {MAX_DF:.0%} degli utenti): {len(eliminate_comuni)} → {eliminate_comuni}")
print(f"Utenti con documento vuoto dopo le soglie: {(np.asarray(X_bow.sum(axis=1)).ravel() == 0).sum()}")

riepilogo = pd.concat([riepilogo, pd.DataFrame([descrivi_matrice(f"BoW (min_df={MIN_DF}, max_df={MAX_DF})", X_bow)])],
                      ignore_index=True)
salva_tabella(riepilogo, "bow_riepilogo")
display(riepilogo)

# Attributi mantenuti: esempi per fascia di diffusione
vocab_info = pd.DataFrame({"parola": vocab,
                           "utenti": np.asarray((X_bow > 0).sum(axis=0)).ravel(),
                           "occorrenze": np.asarray(X_bow.sum(axis=0)).ravel()})
vocab_info["quota utenti [%]"] = (100 * vocab_info["utenti"] / m).round(2)
fasce_diffusione = pd.cut(vocab_info["utenti"], bins=[MIN_DF - 1, 20, 100, 500, 1000, m],
                          labels=[f"{MIN_DF}-20", "21-100", "101-500", "501-1000", f">1000"])
esempi_attributi = (vocab_info.assign(fascia=fasce_diffusione)
                    .groupby("fascia", observed=True)
                    .apply(lambda g: pd.Series({"n. parole": len(g),
                                                "esempi": ", ".join(g.sample(min(12, len(g)), random_state=42)["parola"])}),
                           include_groups=False))
salva_tabella(esempi_attributi, "attributi_mantenuti_esempi", index=True)
with pd.option_context("display.max_colwidth", None):
    display(esempi_attributi)

# **✍️ Risposta 1.4a (da completare con i numeri sopra).** Punti da toccare:
# - la document frequency ha una coda lunghissima: circa metà delle parole compare in un solo utente;
# - `MIN_DF` elimina la maggior parte del vocabolario ma una piccola parte delle occorrenze (tabella `vocab_min`);
# - `MAX_DF` elimina poche parole, quelle usate da più di metà degli utenti (elenco sopra), che non aiutano a
#   distinguere gli utenti;
# - risultato: vocabolario ridotto, stessa informazione utile, meno sparsità e meno rumore.

# ## 4. Salvataggio per i script successivi

utenti[["author", "gender", "n_commenti", "n_token", "fascia_commenti", "doc"]].to_csv(
    OUT / "utenti.csv", index=False, encoding="utf-8")
sparse.save_npz(OUT / "bow.npz", X_bow)
vocab_info.to_csv(OUT / "vocabolario.csv", index=False, encoding="utf-8")
pd.DataFrame([{"MIN_DF": MIN_DF, "MAX_DF": MAX_DF, "vocabolario_completo": len(vocab_full),
               "vocabolario_ridotto": len(vocab)}]).to_csv(TAB / "soglie_scelte.csv", index=False)
print(f"Salvati in {OUT}: utenti.csv, bow.npz ({X_bow.shape[0]:,} × {X_bow.shape[1]:,}), vocabolario.csv")
