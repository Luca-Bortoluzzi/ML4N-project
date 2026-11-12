#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Machine Learning for Networking — Project 7, Fase 1: EDA e qualita' dati.

VERSIONE STATICA: legge direttamente i tre file CSV, senza parametri.
Metti lo script e i CSV nella stessa cartella, quindi esegui:

    python ML4N_Fase1_EDA_Static.py

Oppure cambia CARTELLA_DATI qui sotto con il percorso dei tre file.
L'output viene salvato automaticamente in 'risultati_fase1'.

Richiede: Python >=3.10, pandas, numpy, matplotlib.
Non applica ancora tokenizzazione, stemming, BoW/TF-IDF o PCA/SVD.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Grafici PNG anche da terminale/headless del cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURAZIONE STATICA — MODIFICA QUI SOLO SE I CSV NON SONO ACCANTO
# A QUESTO SCRIPT. Non occorre specificare alcun parametro da terminale.
# ======================================================================
CARTELLA_DATI = Path(__file__).resolve().parent
FILE_SUPERVISED = CARTELLA_DATI / "data_supervised.csv"
FILE_UNSUPERVISED = CARTELLA_DATI / "data_unsupervised.csv"
FILE_TARGET = CARTELLA_DATI / "target_supervised.csv"
CARTELLA_RISULTATI = CARTELLA_DATI / "risultati_fase1"
TOP_N_SUBREDDIT = 20
ESPORTA_DATASET_PULITI = True

COMMENT_COLUMNS = ("author", "subreddit", "created_utc", "body")
FIG_DPI = 155


def read_csv_file(path: Path) -> pd.DataFrame:
    """Legge un singolo CSV e verifica che esista, senza usare ZIP."""
    if not path.is_file():
        raise FileNotFoundError(
            f"File CSV non trovato: {path}\n"
            "Metti i tre CSV nella cartella dello script oppure modifica "
            "CARTELLA_DATI / FILE_* nella configurazione iniziale."
        )
    df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    df.columns = df.columns.str.strip()
    return df


def write_csv(df: pd.DataFrame, out: Path, *, index: bool = False) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=index, encoding="utf-8")


def clean_text_column(df: pd.DataFrame, column: str) -> None:
    # Rimuoviamo solo spazi ai bordi dalle chiavi, mai dal corpo dei commenti.
    df[column] = df[column].astype("string").str.strip()


def validate_comment_schema(df: pd.DataFrame, origin: str) -> None:
    missing = set(COMMENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{origin}: mancano le colonne: {', '.join(sorted(missing))}")


def describe_numeric(values: pd.Series, dataset: str, metric: str) -> dict:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {"dataset": dataset, "metrica": metric, "n": 0}
    q = x.quantile([0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    return {
        "dataset": dataset, "metrica": metric, "n": int(x.size),
        "min": float(x.min()), "media": float(x.mean()),
        "mediana": float(q.loc[0.5]), "p25": float(q.loc[0.25]),
        "p75": float(q.loc[0.75]), "p90": float(q.loc[0.90]),
        "p95": float(q.loc[0.95]), "p99": float(q.loc[0.99]),
        "max": float(x.max())
    }


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_hist_and_ecdf(values: pd.Series, label: str, name: str, x_label: str, figs: Path,
                       *, log_hist_x: bool) -> None:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if x.size == 0:
        return
    plt.figure(figsize=(9, 5.2))
    if log_hist_x:
        pos = x[x > 0]
        top = max(1.01, float(pos.max()))
        bins = np.geomspace(1, top + 1, 50)
        plt.hist(pos, bins=bins, color="#3975ad", edgecolor="white", alpha=0.92)
        plt.xscale("log")
        plt.title(f"{label} — distribuzione (asse X logaritmico)")
    else:
        p99 = float(np.percentile(x, 99))
        upper = max(1.0, p99)
        # La soglia e' usata soltanto per il grafico, NON per rimuovere record.
        plot_values = x[x <= upper]
        plt.hist(plot_values, bins=60, color="#3975ad", edgecolor="white", alpha=0.92)
        plt.title(f"{label} — istogramma (fino al 99° percentile)")
        plt.figtext(0.5, 0.005, f"Valori oltre il 99° percentile esclusi solo dalla figura: {int((x > upper).sum()):,}",
                    ha="center", fontsize=9)
    plt.xlabel(x_label)
    plt.ylabel("Frequenza")
    plt.grid(axis="y", alpha=0.2)
    save_plot(figs / f"{name}_histogram.png")

    unique, counts = np.unique(x, return_counts=True)
    y = np.cumsum(counts, dtype=np.int64) / x.size
    plt.figure(figsize=(9, 5.2))
    plt.step(unique, y, where="post", color="#3975ad", linewidth=1.5)
    if log_hist_x:
        plt.xscale("log")
    plt.xlabel(x_label + (" (scala logaritmica)" if log_hist_x else ""))
    plt.ylabel("Frazione cumulativa degli elementi")
    plt.title(f"{label} — ECDF (distribuzione completa)")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    save_plot(figs / f"{name}_ecdf.png")


def plot_top_subreddits(top: pd.Series, label: str, name: str, figs: Path) -> None:
    if top.empty:
        return
    plt.figure(figsize=(11, max(5.5, 0.32 * len(top) + 1.5)))
    data = top.iloc[::-1]
    plt.barh(data.index.astype(str), data.to_numpy(), color="#4682b4")
    plt.xlabel("Numero di commenti")
    plt.title(f"Subreddit con più commenti — {label}")
    plt.grid(axis="x", alpha=0.2)
    save_plot(figs / f"{name}_top_subreddits.png")


def plot_daily(daily: pd.Series, label: str, name: str, figs: Path) -> None:
    if daily.empty:
        return
    plt.figure(figsize=(11, 4.8))
    plt.plot(daily.index, daily.to_numpy(), marker=".", linewidth=1.2, color="#3975ad")
    plt.ylabel("Numero di commenti")
    plt.xlabel("Data (UTC)")
    plt.title(f"Commenti per giorno — {label}")
    plt.grid(alpha=0.25)
    plt.gcf().autofmt_xdate()
    save_plot(figs / f"{name}_daily_activity.png")


def plot_hours(hourly: pd.Series, label: str, name: str, figs: Path) -> None:
    plt.figure(figsize=(9, 4.7))
    plt.bar(hourly.index.to_numpy(), hourly.to_numpy(), color="#3975ad")
    plt.xticks(np.arange(0, 24, 2))
    plt.xlabel("Ora del giorno (UTC)")
    plt.ylabel("Numero di commenti")
    plt.title(f"Distribuzione oraria — {label}")
    plt.grid(axis="y", alpha=0.2)
    save_plot(figs / f"{name}_hourly_activity.png")


def analyze_comments(name: str, df: pd.DataFrame, out: Path, top_n: int,
                     *, export_clean: bool) -> tuple[pd.DataFrame, dict, pd.Series]:
    """Analizza e pulisce un dataset. Il testo in body rimane invariato."""
    log = logging.getLogger("ML4N")
    validate_comment_schema(df, name)
    n_raw = len(df)
    tables = out / "tables"
    figs = out / "figures"
    log.info("%s: %s righe originali", name, f"{n_raw:,}")

    # Profili prima della pulizia, senza reinterpretare il testo.
    missing = df.loc[:, list(COMMENT_COLUMNS)].isna().sum().astype("int64")
    empty_body = df["body"].notna() & (df["body"].astype("string").str.strip() == "")
    blank_author = df["author"].notna() & (df["author"].astype("string").str.strip() == "")
    blank_subreddit = df["subreddit"].notna() & (df["subreddit"].astype("string").str.strip() == "")
    null_report = pd.DataFrame({
        "campo": list(COMMENT_COLUMNS),
        "valori_null": [int(missing[c]) for c in COMMENT_COLUMNS],
        "valori_vuoti_o_spazi": [int(blank_author.sum()), int(blank_subreddit.sum()), 0, int(empty_body.sum())],
    })
    write_csv(null_report, tables / f"{name}_missing_and_blank.csv")

    duplicate_exact = int(df.duplicated(keep="first").sum())
    duplicate_same_body = int(df.duplicated(subset=["body"], keep="first").sum())
    repeated = df.loc[df.duplicated(keep=False), list(COMMENT_COLUMNS)]
    if not repeated.empty:
        top_dup = (repeated.value_counts(subset=list(COMMENT_COLUMNS), dropna=False)
                   .head(10).rename("n_occorrenze").reset_index())
        top_dup["body_excerpt"] = top_dup["body"].astype("string").str.slice(0, 200)
        top_dup = top_dup.drop(columns=["body"])
        write_csv(top_dup, tables / f"{name}_top_exact_duplicates.csv")

    unique_before = {key: int(df[key].nunique(dropna=True)) for key in COMMENT_COLUMNS}
    write_csv(pd.DataFrame([{"campo": k, "valori_distinti": v} for k, v in unique_before.items()]),
              tables / f"{name}_distinct_values_before.csv")

    # Non confondere doppioni di testo con doppioni di righe complete.
    df = df.drop_duplicates(keep="first").copy()
    clean_text_column(df, "author")
    clean_text_column(df, "subreddit")
    df["body"] = df["body"].astype("string")  # Preserve case, punteggiatura, emoji, stopword.
    author_bad = df["author"].isna() | (df["author"] == "")
    body_bad = df["body"].isna() | (df["body"].str.strip() == "")
    to_remove = author_bad | body_bad
    n_invalid = int(to_remove.sum())
    df = df.loc[~to_remove].copy()
    # Il subreddit mancante non rende il testo inutile: lo conserviamo come NA.

    # Conversione temporale verificabile, senza distruggere il timestamp originale.
    numeric_dates = pd.to_numeric(df["created_utc"], errors="coerce")
    df["created_datetime"] = pd.to_datetime(numeric_dates, unit="s", errors="coerce", utc=True)
    invalid_dates = int(df["created_datetime"].isna().sum())
    df["comment_length"] = df["body"].str.len().astype("int64")
    df["word_count_approx"] = df["body"].str.count(r"\S+").astype("int64")

    # Segnali di contenuti rimossi: conteggiati ma non eliminati di default.
    placeholders = {p: int(df["body"].str.strip().str.lower().eq(p).sum()) for p in ("[deleted]", "[removed]")}

    user_counts = df.groupby("author", dropna=False, observed=True).size().sort_values(ascending=False)
    top_sub = df["subreddit"].value_counts(dropna=True).head(top_n)
    sub_counts = df.groupby("subreddit", dropna=False, observed=True).agg(
        n_commenti=("author", "size"), n_autori=("author", "nunique")
    ).sort_values("n_commenti", ascending=False).reset_index()
    top_share = (100.0 * top_sub.sum() / len(df)) if len(df) else 0.0

    write_csv(user_counts.rename("n_commenti").reset_index(), tables / f"{name}_comments_per_author.csv")
    write_csv(sub_counts, tables / f"{name}_subreddits_counts_and_authors.csv")
    write_csv(pd.DataFrame([describe_numeric(df["comment_length"], name, "caratteri"),
                            describe_numeric(df["word_count_approx"], name, "parole_approssimate"),
                            describe_numeric(user_counts, name, "commenti_per_autore")]),
              tables / f"{name}_numeric_summary.csv")

    daily = df.dropna(subset=["created_datetime"]).groupby(df["created_datetime"].dt.floor("D")).size().sort_index()
    hourly = df["created_datetime"].dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index()
    write_csv(daily.rename("n_commenti").rename_axis("data_utc").reset_index(),
              tables / f"{name}_daily_counts.csv")
    write_csv(hourly.rename("n_commenti").rename_axis("ora_utc").reset_index(),
              tables / f"{name}_hourly_counts.csv")

    plot_hist_and_ecdf(df["comment_length"], f"Lunghezza commenti — {name}",
                       f"{name}_length", "Caratteri per commento", figs, log_hist_x=False)
    plot_hist_and_ecdf(user_counts, f"Commenti per autore — {name}",
                       f"{name}_authors", "Numero di commenti per autore", figs, log_hist_x=True)
    plot_top_subreddits(top_sub, name, name, figs)
    plot_daily(daily, name, name, figs)
    plot_hours(hourly, name, name, figs)

    if export_clean:
        # CSV GZIP compatibile con pandas anche in assenza di pyarrow.
        # Manteniamo soltanto le colonne originarie e il timestamp ISO per la Fase 2.
        keep = [*COMMENT_COLUMNS, "created_datetime"]
        destination = out / "data_clean" / f"data_{name}_clean.csv.gz"
        log.info("Salvo dataset pulito: %s", destination)
        df[keep].to_csv(destination, index=False, compression="gzip", encoding="utf-8")

    timestamp_min = df["created_datetime"].min()
    timestamp_max = df["created_datetime"].max()
    summary = {
        "dataset": name,
        "righe_originali": n_raw, "duplicati_esatti_rimossi": duplicate_exact,
        "testi_ripetuti_su_righe_originali": duplicate_same_body,
        "righe_non_valide_rimosse": n_invalid, "righe_pulite": len(df),
        "autori_distinti": df["author"].nunique(dropna=True),
        "subreddit_distinti": df["subreddit"].nunique(dropna=True),
        "timestamp_distinti": df["created_utc"].nunique(dropna=True),
        "testi_distinti": df["body"].nunique(dropna=True),
        "testi_null_originali": int(missing["body"]),
        "timestamp_non_validi": invalid_dates,
        "data_inizio_utc": timestamp_min.isoformat() if pd.notna(timestamp_min) else None,
        "data_fine_utc": timestamp_max.isoformat() if pd.notna(timestamp_max) else None,
        "quota_commenti_top_subreddit_pct": round(top_share, 3),
        "n_commenti_deleted": placeholders["[deleted]"],
        "n_commenti_removed": placeholders["[removed]"],
        "lunghezza_media": round(float(df["comment_length"].mean()), 3) if len(df) else None,
        "lunghezza_mediana": round(float(df["comment_length"].median()), 3) if len(df) else None,
        "commenti_per_autore_media": round(float(user_counts.mean()), 3) if len(user_counts) else None,
        "commenti_per_autore_mediana": round(float(user_counts.median()), 3) if len(user_counts) else None,
    }
    log.info("%s: %s righe pulite; %s duplicati; %s record senza testo/autore",
             name, f"{len(df):,}", f"{duplicate_exact:,}", f"{n_invalid:,}")
    # Evitiamo di tenere in memoria 1.1 milioni di commenti insieme ai due dataset.
    # Ritorniamo il dataframe solo per la supervised (necessario a controlli per genere).
    return df, summary, user_counts


def analyze_target(target: pd.DataFrame, df_supervised: pd.DataFrame, counts: pd.Series,
                   unsupervised_author_set: set[str], out: Path) -> tuple[dict, list[dict]]:
    if not {"author", "gender"}.issubset(target.columns):
        raise ValueError("target_supervised.csv: colonne 'author' e 'gender' obbligatorie")
    target = target[["author", "gender"]].copy()
    clean_text_column(target, "author")
    labels_numeric = pd.to_numeric(target["gender"], errors="coerce")
    n_invalid_labels = int((labels_numeric.isna() | ~labels_numeric.isin([0, 1])).sum())
    target["gender"] = labels_numeric
    n_duplicate_author = int(target["author"].duplicated().sum())
    conflict = int((target.groupby("author", dropna=False)["gender"].nunique(dropna=True) > 1).sum())
    if n_invalid_labels or conflict or target["author"].isna().any() or (target["author"] == "").any():
        raise ValueError("Target contiene etichette o autori non validi: correggere prima di continuare")
    target = target.drop_duplicates("author", keep="first").copy()
    target["gender"] = target["gender"].astype("int8")

    s_authors = set(df_supervised["author"].dropna().tolist())
    t_authors = set(target["author"].tolist())
    missing_labels = s_authors - t_authors
    label_without_comments = t_authors - s_authors
    overlap = s_authors & unsupervised_author_set
    if missing_labels:
        raise ValueError(f"Mancano etichette per {len(missing_labels)} autori supervised. Esempi: {sorted(missing_labels)[:5]}")
    checks = [
        {"controllo": "Etichette non valide", "valore": n_invalid_labels, "esito": "OK" if n_invalid_labels == 0 else "ATTENZIONE"},
        {"controllo": "Autori duplicati nel target", "valore": n_duplicate_author, "esito": "OK" if n_duplicate_author == 0 else "ATTENZIONE"},
        {"controllo": "Autori con etichette conflittuali", "valore": conflict, "esito": "OK" if conflict == 0 else "ERRORE"},
        {"controllo": "Autori supervised senza target", "valore": len(missing_labels), "esito": "OK" if not missing_labels else "ERRORE"},
        {"controllo": "Target senza commenti supervised", "valore": len(label_without_comments), "esito": "OK" if not label_without_comments else "ATTENZIONE"},
        {"controllo": "Autori in entrambi i dataset", "valore": len(overlap), "esito": "OK" if not overlap else "ATTENZIONE"},
    ]
    write_csv(pd.DataFrame(checks), out / "tables" / "integrity_checks.csv")
    write_csv(target, out / "data_clean" / "target_supervised_checked.csv")

    gender_count = target["gender"].value_counts().reindex([0, 1], fill_value=0).rename("n_utenti")
    dist = gender_count.rename_axis("gender").reset_index()
    dist["label"] = dist["gender"].map({0: "Maschile (0)", 1: "Femminile (1)"})
    dist["percentuale"] = (dist["n_utenti"] / dist["n_utenti"].sum() * 100).round(4)
    write_csv(dist, out / "tables" / "gender_distribution.csv")
    plt.figure(figsize=(7, 4.6))
    colors = ["#3975ad", "#a779b5"]
    plt.bar(dist["label"], dist["n_utenti"], color=colors)
    for i, val in enumerate(dist["n_utenti"]):
        plt.text(i, val + max(8, val * 0.012), f"{val:,} ({dist.loc[i, 'percentuale']:.2f}%)", ha="center")
    plt.ylim(0, max(dist["n_utenti"]) * 1.17)
    plt.ylabel("Numero di utenti")
    plt.title("Etichette di genere — utenti supervised")
    plt.grid(axis="y", alpha=0.2)
    save_plot(out / "figures" / "gender_distribution.png")

    author_stats = counts.rename("n_commenti").reset_index().merge(target, on="author", how="left", validate="one_to_one")
    if author_stats["gender"].isna().any():
        raise RuntimeError("Join target-autori incompleto (controllo interno)")
    author_stats["gender"] = author_stats["gender"].astype("int8")
    write_csv(author_stats, out / "tables" / "supervised_comments_per_author_and_gender.csv")
    by_gender = author_stats.groupby("gender")["n_commenti"].agg(["count", "sum", "mean", "median", "min", "max"]).reset_index()
    by_gender = by_gender.rename(columns={"count":"n_utenti", "sum":"n_commenti", "mean":"commenti_media", "median":"commenti_mediana", "min":"commenti_min", "max":"commenti_max"})
    write_csv(by_gender, out / "tables" / "comments_by_gender.csv")
    plt.figure(figsize=(7.5, 5))
    data = [author_stats.loc[author_stats["gender"] == g, "n_commenti"].to_numpy() for g in [0, 1]]
    plt.boxplot(data, showfliers=False)
    plt.xticks([1, 2], ["Maschile (0)", "Femminile (1)"])
    plt.yscale("log")
    plt.ylabel("Commenti per autore (scala log)")
    plt.title("Volume commenti per etichetta (outlier non disegnati)")
    plt.grid(axis="y", alpha=0.25)
    save_plot(out / "figures" / "comments_by_gender_boxplot.png")

    target_summary = {
        "utenti_target": len(target), "utenti_classe_0": int(gender_count.loc[0]),
        "utenti_classe_1": int(gender_count.loc[1]),
        "quota_classe_0_pct": round(float(dist.loc[0, "percentuale"]), 3),
        "quota_classe_1_pct": round(float(dist.loc[1, "percentuale"]), 3),
        "overlap_supervised_unsupervised": len(overlap)
    }
    return target_summary, checks


def report_markdown(out: Path, summaries: list[dict], t: dict, checks: list[dict]) -> None:
    def it_number(num: object) -> str:
        if num is None:
            return "n.d."
        if isinstance(num, (float, np.floating)):
            return f"{num:,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")
        return f"{int(num):,}".replace(",", ".")

    lines = [
        "# ML4N Project 7 — Fase 1: esplorazione e qualità dei dati", "",
        "## Ambito", "",
        "Il report copre le analisi esplorative (EDA), la verifica delle etichette, la gestione delle righe duplicate",
        "e la preparazione dei CSV. Il testo dei commenti viene conservato nella forma originale:",
        "tokenizzazione, stopword removal, lemmatizzazione, BoW, TF-IDF e PCA/SVD saranno svolti in una fase successiva.", "",
        "## Riepilogo dataset", "",
        "| Dataset | Righe originali | Duplicati esatti | Altre righe rimosse | Righe pulite | Autori | Subreddit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append("| {dataset} | {a} | {b} | {c} | {d} | {e} | {f} |".format(
            dataset=s["dataset"], a=it_number(s["righe_originali"]), b=it_number(s["duplicati_esatti_rimossi"]),
            c=it_number(s["righe_non_valide_rimosse"]), d=it_number(s["righe_pulite"]),
            e=it_number(s["autori_distinti"]), f=it_number(s["subreddit_distinti"])))
    lines += ["", "## Indicatori descrittivi sui dati puliti", ""]
    for s in summaries:
        lines += [f"### {s['dataset'].capitalize()}", "",
                  f"- Lunghezza media del testo: {it_number(s['lunghezza_media'])} caratteri.",
                  f"- Lunghezza mediana: {it_number(s['lunghezza_mediana'])} caratteri.",
                  f"- Commenti medi per autore: {it_number(s['commenti_per_autore_media'])}.",
                  f"- Commenti mediani per autore: {it_number(s['commenti_per_autore_mediana'])}.",
                  f"- Intervallo dei dati (UTC): {s['data_inizio_utc']} — {s['data_fine_utc']}.",
                  f"- Timestamp non validi: {it_number(s['timestamp_non_validi'])}.",
                  f"- Commenti '[deleted]': {it_number(s['n_commenti_deleted'])}; '[removed]': {it_number(s['n_commenti_removed'])}.",
                  f"- Quota dei commenti nei subreddit principali: {it_number(s['quota_commenti_top_subreddit_pct'])}%.", ""]
    lines += ["## Classi supervisionate", "",
              f"- Utenti etichettati: {it_number(t['utenti_target'])}.",
              f"- Classe 0 (maschile): {it_number(t['utenti_classe_0'])} ({it_number(t['quota_classe_0_pct'])}%).",
              f"- Classe 1 (femminile): {it_number(t['utenti_classe_1'])} ({it_number(t['quota_classe_1_pct'])}%).",
              "- Le etichette sono auto-dichiarate e non costituiscono una verifica indipendente dell'identità di genere.", "",
              "## Controlli di integrità", "",
              "| Controllo | Valore | Esito |", "|---|---:|---|" ]
    for c in checks:
        lines.append(f"| {c['controllo']} | {it_number(c['valore'])} | {c['esito']} |")
    lines += ["", "## Interpretazione e cautele", "",
              "- I duplicati esatti sono eliminati; i testi identici con metadati differenti non sono automaticamente cancellati.",
              "- La distribuzione dei commenti per autore può essere molto sbilanciata: non confondere percentuali per commento e per utente.",
              "- Negli istogrammi della lunghezza il 99° percentile è un limite **grafico**, mentre le ECDF includono tutti i dati.",
              "- I grafici temporali usano UTC, non l'ora locale.",
              "- `[deleted]` e `[removed]` sono segnalati ma mantenuti: decidere e documentare una politica nella fase successiva.",
              "- Il testo rimane originale anche per consentire una corretta analisi VADER nella fase Sentiment Analysis.",
              "- L'EDA non stabilisce una soglia ottimale per modelli ML e non produce ancora vettori numerici.", "",
              "## File generati", "",
              "- `tables/`: statistiche, frequenze, distribuzioni, controllo etichette e integrità.",
              "- `figures/`: istogrammi, ECDF, subreddit, grafici temporali e confronto tra classi.",
              "- `data_clean/`: dati ripuliti dai duplicati esatti (se l'esportazione è abilitata).", ""]
    (out / "report_fase1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if TOP_N_SUBREDDIT < 1:
        raise ValueError("TOP_N_SUBREDDIT deve essere >= 1")
    # Controlliamo tutti e tre i file prima di creare output.
    for path in (FILE_SUPERVISED, FILE_UNSUPERVISED, FILE_TARGET):
        if not path.is_file():
            raise FileNotFoundError(
                f"Manca il file {path.name} nella cartella {path.parent}. "
                "Metti qui i tre CSV, oppure modifica i percorsi "
                "FILE_SUPERVISED / FILE_UNSUPERVISED / FILE_TARGET."
            )

    out = CARTELLA_RISULTATI.expanduser().resolve()
    for p in (out / "tables", out / "figures", out / "data_clean"):
        p.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "esecuzione.log", encoding="utf-8")])
    log = logging.getLogger("ML4N")
    log.info("Analisi Fase 1 — lettura dei tre file CSV dalla cartella: %s", CARTELLA_DATI)
    log.info("Risultati in: %s", out)

    target = read_csv_file(FILE_TARGET)
    df_supervised, summary_s, counts_s = analyze_comments(
        "supervised", read_csv_file(FILE_SUPERVISED), out, TOP_N_SUBREDDIT,
        export_clean=ESPORTA_DATASET_PULITI)
    # Conserviamo solo gli autori del dataset unsupervised dopo il riepilogo.
    df_unsupervised, summary_u, _ = analyze_comments(
        "unsupervised", read_csv_file(FILE_UNSUPERVISED), out, TOP_N_SUBREDDIT,
        export_clean=ESPORTA_DATASET_PULITI)
    unsup_authors = set(df_unsupervised["author"].dropna().tolist())
    del df_unsupervised
    label_summary, checks = analyze_target(target, df_supervised, counts_s,
                                           unsup_authors, out)
    write_csv(pd.DataFrame([summary_s, summary_u]), out / "tables" / "dataset_summary.csv")
    report_markdown(out, [summary_s, summary_u], label_summary, checks)
    log.info("FINE. Report: %s", out / "report_fase1.md")
    log.info("Grafici in %s; tabelle in %s", out / "figures", out / "tables")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Analisi interrotta da un errore")
        raise
