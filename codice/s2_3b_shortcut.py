#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML4N Project 7 — Sezione 2.3b: togliere le scorciatoie e vedere cosa resta.

Due scorciatoie possibili per indovinare il genere:
  1. commenti scritti in subreddit molto sbilanciati verso un genere (MakeupAddiction, nfl, ...);
  2. commenti in cui l'autore rivela il genere ("as a guy", "I'm a woman", "my husband", "[25M]").
Si tolgono questi commenti dal dataset (training E test), si ricostruisce il documento di ogni utente e si
ripetono addestramento e valutazione in quattro scenari: nessuna rimozione, senza 1, senza 2, senza entrambi.

Riferimenti alla teoria (S = pacco di slide, pagina del PDF; Note = note del corso):
  - Criterio per i subreddit sbilanciati = regola di Bayes [S10 p. 30-31; Note p. 75]. Per ogni subreddit s,
    x_s = "l'utente ha scritto almeno un commento in s". Sul SOLO training si stimano per frequenza relativa
    (massima verosimiglianza [S10 p. 27, 31]) la probabilita' a priori P(F) e le verosimiglianze P(x_s=1|F),
    P(x_s=1|M), con lo smoothing di Laplace (+1 a ogni conteggio) per evitare probabilita' nulle [Note p. 76].
    In forma di odds la regola di Bayes e':  P(F|x_s=1)/P(M|x_s=1) = [P(F)/P(M)] * LR_s,
    con LR_s = P(x_s=1|F)/P(x_s=1|M).  Il classificatore di Bayes sceglie la classe con la probabilita' a
    posteriori piu' alta [S10 p. 30]: senza altre informazioni sceglie "maschile" (P(F) = 27%).
    Un subreddit e' sbilanciato verso F se basta da solo a cambiare la decisione, cioe' P(F|x_s=1) > 1/2,
    ovvero LR_s > P(M)/P(F) (circa 2,7); verso M con la soglia simmetrica LR_s < P(F)/P(M).
    Si considerano solo i subreddit con almeno MIN_DF utenti di training, la stessa soglia usata per le
    parole rare (1.4a): sotto, la stima per frequenza poggia su troppo pochi utenti.
  - Commenti espliciti: espressioni regolari sul testo ORIGINALE (colonna body), prima del preprocessing,
    perche' lo stemming e le stopword cancellano proprio "I'm", "a", "my". La precisione delle regole,
    TP/(TP+FP) [S8 p. 18; Note p. 54], si stima controllando a mano un campione casuale dei commenti trovati
    (tables/commenti_espliciti_campione.csv).
  - Pipeline: si ricostruiscono vocabolario, idf e modello sul nuovo training [S7 p. 34, 44]; il test si usa
    una volta per scenario, alla fine [S7 p. 53-54]. Le scelte che imparano dai dati (quali subreddit sono
    sbilanciati) usano solo gli utenti di training [S7 p. 34].
  - Gli iperparametri restano quelli della pipeline scelta nel punto 2.2c: si cambia un solo fattore, i dati,
    cosi' la differenza di macro-F1 misura solo l'effetto della rimozione. La CV a 5 fold stratificata sul nuovo
    training [S7 p. 37-41, 47] da' media e dev. std, per capire se una differenza supera la variabilita'.
  - Cosa impara il modello: le parole con il peso maggiore. In un modello lineare il peso dipende dall'unita'
    di misura della feature [S10 p. 12]; la TF-IDF non e' standardizzata, quindi si confronta
    peso * deviazione standard della feature, che e' il peso che si avrebbe sulla feature standardizzata [S3 p. 59].
  - Metriche: macro-F1, accuracy, F1 femminile [S8 p. 18-20; Note p. 55].

Esecuzione, dalla cartella codice/ (dopo s2_2c_tuning.py):
    python s2_3b_shortcut.py
"""

from __future__ import annotations

import logging
import re
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from comune import FASCE, FILE_COMMENTI, MIN_DF, RISULTATI, SEED, carica_split

TAB, FIG = RISULTATI / "tables", RISULTATI / "figures"
FILE_MODELLO = RISULTATI / "pipeline_migliore.joblib"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
N_JOBS = -1
MIN_UTENTI_SUBREDDIT = MIN_DF          # stessa soglia delle parole rare (1.4a)
N_CAMPIONE = 60                        # commenti espliciti da controllare a mano
N_PAROLE = 20                          # parole con il peso maggiore per ciascun verso
log = logging.getLogger("ML4N")

# Espressioni regolari sul testo originale, senza distinguere maiuscole e minuscole.
NOMI = r"(?:man|woman|guy|girl|dude|lady|gal|male|female|mom|mum|mother|dad|father|wife|husband|boyfriend|girlfriend)"
AGGETTIVI = r"(?:(?:straight|gay|single|young|old|married|tall|short|black|white|bi|trans|grown|middle-aged)\s+)?"
# "as a guy" vale solo se segue la prima persona ("as a guy, I...", "as a woman who...", "as a man am"):
# esclude "as a guy once told me", "his role as a father", "as a girl, you might..." (dal controllo a mano).
DOPO_AS = r"(?:\s*,\s*(?:i|i'm|i've|i'd|im|my|me|we)\b|\s+(?:who|myself|i|am|here)\b|\s*\.\.\.)"
REGOLE = {
    # "I'm a guy", "I am a woman", "I'm female", "as a mom, I", "as a straight male who"
    "dichiarazione": rf"\b(?:i\s*'?\s*m|i\s+am)\s+an?\s+{AGGETTIVI}{NOMI}\b|\bi\s*'?\s*m\s+(?:fe)?male\b"
                     rf"|\bi\s+am\s+(?:fe)?male\b|\bas\s+an?\s+{AGGETTIVI}{NOMI}{DOPO_AS}",
    # "my husband", "my gf", "my fiancee": il genere del partner suggerisce quello dell'autore.
    # "my BF" seguito da un numero o da "at/around" e' la percentuale di grasso corporeo (body fat), non il partner.
    "partner": r"\bmy\s+(?:husband|wife|hubby|boyfriend|girlfriend|gf|fianc[eé]e?|bf(?!\s+(?:at|around|of)\b)(?!\s*~?\d))\b",
    # tag di Reddit con eta' e genere: "[25M]", "(F29)", "(21/f)"; non preceduti da '-' o cifra (temperature)
    "tag eta-genere": r"(?<![-\d])[\(\[]\s*(?:1[3-9]|[2-6]\d)\s*/?\s*[mf]\s*[\)\]]|[\(\[]\s*[mf]\s*/?\s*(?:1[3-9]|[2-6]\d)\s*[\)\]]",
}
SCENARI = ["nessuna rimozione", "senza subreddit sbilanciati", "senza commenti espliciti", "senza entrambi",
           "controllo: stessi commenti tolti a caso"]


def md(df: pd.DataFrame) -> str:
    """Tabella Markdown senza dipendenze esterne."""
    righe = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    righe += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(righe)


def subreddit_sbilanciati(commenti: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Regola di Bayes con smoothing di Laplace, stimata sui soli utenti di training."""
    genere = dict(zip(train["author"], train["gender"]))
    presenze = commenti.loc[commenti["author"].isin(genere), ["author", "subreddit"]].drop_duplicates()
    presenze["F"] = presenze["author"].map(genere)
    s = presenze.groupby("subreddit")["F"].agg(utenti="size", femminili="sum")
    s["maschili"] = s["utenti"] - s["femminili"]
    n_f = int(train["gender"].sum()); n_m = len(train) - n_f
    s["P(s|F)"] = (s["femminili"] + 1) / (n_f + 2)            # Laplace: x_s binaria, +1 su ciascuno dei 2 valori
    s["P(s|M)"] = (s["maschili"] + 1) / (n_m + 2)
    s["LR"] = s["P(s|F)"] / s["P(s|M)"]
    odds_prior = n_f / n_m
    s["P(F|s)"] = s["LR"] * odds_prior / (1 + s["LR"] * odds_prior)
    soglia = 1 / odds_prior                                   # LR oltre cui P(F|s) > 1/2
    s["verso"] = np.where(s["LR"] > soglia, "femminile", np.where(s["LR"] < 1 / soglia, "maschile", ""))
    s.loc[s["utenti"] < MIN_UTENTI_SUBREDDIT, "verso"] = ""
    s = s.sort_values("LR", ascending=False).reset_index()
    for c in ["P(s|F)", "P(s|M)", "LR", "P(F|s)"]:
        s[c] = s[c].round(4)
    return s, soglia


def commenti_espliciti(commenti: pd.DataFrame) -> pd.DataFrame:
    """Una colonna booleana per regola, piu' 'esplicito' = almeno una regola."""
    out = pd.DataFrame(index=commenti.index)
    for nome, regola in REGOLE.items():
        out[nome] = commenti["body"].str.contains(regola, flags=re.IGNORECASE, regex=True)
    out["esplicito"] = out[list(REGOLE)].any(axis=1)
    return out


def campione_da_controllare(commenti: pd.DataFrame, espl: pd.DataFrame) -> pd.DataFrame:
    """Campione casuale dei commenti trovati, con il pezzo di testo che ha fatto scattare la regola."""
    idx = espl.index[espl["esplicito"]]
    # Le regole sono state rifinite guardando un primo campione (seme SEED): la precisione si stima su un campione
    # NUOVO (seme SEED + 1), come non si valuta un modello sui dati con cui e' stato costruito [S7 p. 50, 53].
    scelti = pd.Series(idx).sample(n=min(N_CAMPIONE, len(idx)), random_state=SEED + 1)
    righe = []
    for i in scelti:
        testo = commenti.at[i, "body"]
        for nome, regola in REGOLE.items():
            m = re.search(regola, testo, flags=re.IGNORECASE)
            if m:
                righe.append({"regola": nome, "trovato": m.group(0),
                              "contesto": testo[max(0, m.start() - 80): m.end() + 80].replace("\n", " "),
                              "corretto (da compilare a mano)": ""})
                break
    return pd.DataFrame(righe)


def controllo_casuale(commenti: pd.DataFrame, tieni_entrambi: pd.Series) -> pd.Series:
    """Ogni utente perde lo STESSO numero di commenti che perde in "senza entrambi", ma scelti a caso.

    Togliere commenti riduce anche la quantita' di testo, che da sola abbassa la macro-F1 (2.2d). Il confronto
    con questo scenario separa l'effetto delle scorciatoie da quello della quantita' di testo.
    """
    rng = np.random.default_rng(SEED)
    tieni = pd.Series(True, index=commenti.index)
    tolti = (~tieni_entrambi).groupby(commenti["author"]).sum()
    for autore, idx in commenti.groupby("author").groups.items():
        k = int(tolti.get(autore, 0))
        if k:
            tieni.loc[rng.choice(np.asarray(idx), size=k, replace=False)] = False
    return tieni


def documenti(commenti: pd.DataFrame, tieni: pd.Series) -> pd.Series:
    """Stesso documento della Sezione 1: commenti non vuoti dell'utente uniti da uno spazio."""
    k = commenti[tieni]
    return k.groupby("author")["clean_text"].agg(lambda s: " ".join(t for t in s if t))


def pesi_parole(pipe, docs_train: pd.Series) -> pd.DataFrame:
    """Peso * dev. std della feature sul training = peso sulla feature standardizzata [S10 p. 12; S3 p. 59]."""
    X = pipe[:-1].transform(docs_train)
    media = np.asarray(X.mean(axis=0)).ravel()
    media2 = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    std = np.sqrt(np.maximum(media2 - media ** 2, 0))
    w = pipe[-1].coef_.ravel() * std
    nomi = pipe.named_steps["bow"].get_feature_names_out()
    # Quota delle occorrenze della parola dovuta al solo utente che la usa di piu': se e' alta, il peso riflette
    # un singolo utente estremo (outlier [S3 p. 56]) e non un'abitudine di molti utenti.
    conteggi = pipe.named_steps["bow"].transform(docs_train).tocsc()
    somma = np.asarray(conteggi.sum(axis=0)).ravel()
    massimo = conteggi.max(axis=0).toarray().ravel()
    quota = np.divide(massimo, somma, out=np.zeros_like(somma, dtype=float), where=somma > 0)
    ordine = np.argsort(w)
    f, m = ordine[::-1][:N_PAROLE], ordine[:N_PAROLE]
    return pd.DataFrame({"verso femminile": nomi[f], "peso F": w[f].round(4), "quota 1 utente F": quota[f].round(2),
                         "verso maschile": nomi[m], "peso M": w[m].round(4), "quota 1 utente M": quota[m].round(2)})


def esegui_scenario(nome: str, base, commenti, tieni, train, test) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    docs = documenti(commenti, tieni)
    tr = train[train["author"].isin(docs.index)].copy()
    te = test[test["author"].isin(docs.index)].copy()
    tr["doc"] = tr["author"].map(docs); te["doc"] = te["author"].map(docs)

    cv = cross_validate(clone(base), tr["doc"], tr["gender"], scoring="f1_macro", cv=CV, n_jobs=N_JOBS)
    pipe = clone(base)
    t0 = time.perf_counter()
    pipe.fit(tr["doc"], tr["gender"])
    tempo = time.perf_counter() - t0
    te["pred"] = pipe.predict(te["doc"])
    modello = pipe[-1]
    iterazioni = int(np.max(modello.n_iter_))
    riga = {"scenario": nome,
            "commenti tolti": int((~tieni).sum()), "quota commenti tolti": round(float((~tieni).mean()), 4),
            "utenti training": len(tr), "utenti test": len(te),
            "macro-F1 CV": round(cv["test_score"].mean(), 4), "dev. std CV": round(cv["test_score"].std(), 4),
            "macro-F1 test": round(f1_score(te["gender"], te["pred"], average="macro"), 4),
            "accuracy test": round(accuracy_score(te["gender"], te["pred"]), 4),
            "F1 femminile test": round(f1_score(te["gender"], te["pred"], pos_label=1), 4),
            "tempo training [s]": round(tempo, 2), "iterazioni": iterazioni,
            "converge": "si'" if iterazioni < modello.max_iter else "NO"}
    log.info("%s: %s", nome, riga)
    return riga, te[["author", "gender", "fascia_commenti", "pred"]].assign(scenario=nome), pesi_parole(pipe, tr["doc"])


def grafico(fasce: pd.DataFrame) -> None:
    colori = ["#7f7f7f", "#2f6fd6", "#eb6834", "#1baa7a", "#c9b037"]
    gruppi = FASCE + ["tutti"]
    x = np.arange(len(gruppi)); larg = 0.8 / len(SCENARI)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for k, (s, c) in enumerate(zip(SCENARI, colori)):
        v = [fasce.loc[(fasce["scenario"] == s) & (fasce["fascia"] == g), "macro-F1"].iloc[0] for g in gruppi]
        pos = x + (k - (len(SCENARI) - 1) / 2) * larg
        ax.bar(pos, v, width=larg * 0.92, color=c, label=s)
        for xi, vi in zip(pos, v):
            ax.annotate(f"{vi:.2f}", (xi, vi), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(x, gruppi); ax.set_ylim(0.3, 1.0); ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Commenti per utente (numero originale, utenti di test comuni a tutti gli scenari)")
    ax.set_ylabel("macro-F1 sul test")
    ax.set_title("Quanto scende la macro-F1 togliendo le scorciatoie (pipeline del 2.2c)")
    ax.legend(fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    plt.tight_layout(); plt.savefig(FIG / "shortcut_scenari.png", dpi=150, bbox_inches="tight"); plt.close()


def main() -> int:
    for p in (TAB, FIG):
        p.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(RISULTATI / "esecuzione_s2_3b.log", encoding="utf-8")])
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)
    if not FILE_MODELLO.is_file():
        raise FileNotFoundError(f"Manca {FILE_MODELLO}: eseguire prima s2_2c_tuning.py")
    base = joblib.load(FILE_MODELLO)            # solo gli iperparametri: ogni scenario usa clone() = non addestrata

    train, test = carica_split()
    commenti = pd.read_csv(FILE_COMMENTI, keep_default_na=False, usecols=["author", "subreddit", "body", "clean_text"])
    commenti = commenti[commenti["author"].isin(set(train["author"]) | set(test["author"]))].reset_index(drop=True)
    log.info("Commenti degli utenti dello split: %d", len(commenti))

    # 1. Subreddit sbilanciati (solo training)
    sub, soglia = subreddit_sbilanciati(commenti, train)
    sub.to_csv(TAB / "subreddit_sbilanciati_tutti.csv", index=False)
    scelti = sub[sub["verso"] != ""]
    scelti.to_csv(TAB / "subreddit_sbilanciati.csv", index=False)
    nel_sbil = commenti["subreddit"].isin(set(scelti["subreddit"]))
    log.info("Subreddit sbilanciati: %d (F %d, M %d), soglia LR %.3f; commenti coinvolti %.1f%%", len(scelti),
             (scelti["verso"] == "femminile").sum(), (scelti["verso"] == "maschile").sum(), soglia, 100 * nel_sbil.mean())

    # 2. Commenti espliciti
    espl = commenti_espliciti(commenti)
    conteggi = pd.DataFrame([{"regola": r, "commenti": int(espl[r].sum()),
                              "quota commenti": round(float(espl[r].mean()), 4),
                              "utenti": int(commenti.loc[espl[r], "author"].nunique())}
                             for r in list(REGOLE) + ["esplicito"]])
    conteggi["regola"] = conteggi["regola"].replace({"esplicito": "almeno una regola"})
    conteggi.to_csv(TAB / "commenti_espliciti_conteggi.csv", index=False)
    campione_da_controllare(commenti, espl).to_csv(TAB / "commenti_espliciti_campione.csv", index=False)

    # 3. Scenari
    tieni = {"nessuna rimozione": pd.Series(True, index=commenti.index),
             "senza subreddit sbilanciati": ~nel_sbil,
             "senza commenti espliciti": ~espl["esplicito"],
             "senza entrambi": ~nel_sbil & ~espl["esplicito"]}
    tieni["controllo: stessi commenti tolti a caso"] = controllo_casuale(commenti, tieni["senza entrambi"])
    righe, pred, pesi = [], [], {}
    for s in SCENARI:
        r, p, w = esegui_scenario(s, base, commenti, tieni[s], train, test)
        righe.append(r); pred.append(p); pesi[s] = w
    confronto = pd.DataFrame(righe)
    base_f1 = confronto.loc[0, "macro-F1 test"]
    confronto["differenza test"] = (confronto["macro-F1 test"] - base_f1).round(4)

    # Confronto equo: solo gli utenti di test presenti in tutti gli scenari, fasce dal numero ORIGINALE di commenti
    pred = pd.concat(pred)
    comuni = set.intersection(*[set(g["author"]) for _, g in pred.groupby("scenario")])
    pc = pred[pred["author"].isin(comuni)]
    fasce = []
    for s in SCENARI:
        g0 = pc[pc["scenario"] == s]
        for f in FASCE + ["tutti"]:
            g = g0 if f == "tutti" else g0[g0["fascia_commenti"].astype(str) == f]
            fasce.append({"scenario": s, "fascia": f, "utenti": len(g), "femminile": int(g["gender"].sum()),
                          "macro-F1": round(f1_score(g["gender"], g["pred"], average="macro"), 3)})
    fasce = pd.DataFrame(fasce)
    confronto["macro-F1 test (utenti comuni)"] = [fasce.loc[(fasce["scenario"] == s) & (fasce["fascia"] == "tutti"),
                                                            "macro-F1"].iloc[0] for s in SCENARI]
    confronto.to_csv(TAB / "shortcut_confronto.csv", index=False)
    fasce.to_csv(TAB / "shortcut_per_fascia.csv", index=False)
    for s, w in pesi.items():
        w.to_csv(TAB / f"shortcut_pesi_{re.sub(r'[^a-z0-9]+', '_', s).strip('_')}.csv", index=False)
    grafico(fasce)

    log.info("\n%s", confronto.to_string(index=False))
    log.info("\n%s", fasce.pivot(index="fascia", columns="scenario", values="macro-F1").to_string())

    tab_sub = scelti[["subreddit", "verso", "utenti", "femminili", "LR", "P(F|s)"]]
    testo = f"""# ML4N Project 7 — Sezione 2.3b: rimozione delle scorciatoie

## 1. Subreddit sbilanciati (stimati sui soli utenti di training)
Regola di Bayes con smoothing di Laplace: un subreddit e' sbilanciato verso F se, da solo, porta la probabilita'
a posteriori P(F|s) sopra 1/2, cioe' LR = P(s|F)/P(s|M) > P(M)/P(F) = {soglia:.3f}; verso M se LR < {1/soglia:.3f}.
Solo subreddit con almeno {MIN_UTENTI_SUBREDDIT} utenti di training.
Subreddit sbilanciati: {len(scelti)} ({(scelti['verso'] == 'femminile').sum()} verso F,
{(scelti['verso'] == 'maschile').sum()} verso M); i loro commenti sono il {100 * nel_sbil.mean():.1f}% del totale.

Verso F (primi 15):
{md(tab_sub[tab_sub['verso'] == 'femminile'].head(15))}

Verso M (primi 15):
{md(tab_sub[tab_sub['verso'] == 'maschile'].sort_values('LR').head(15))}

Elenco completo: `tables/subreddit_sbilanciati.csv`.

## 2. Commenti in cui l'autore rivela il genere
Espressioni regolari sul testo originale (maiuscole e minuscole indifferenti):
{chr(10).join(f'- **{k}**: `{v}`' for k, v in REGOLE.items())}

{md(conteggi)}

Campione casuale di {N_CAMPIONE} commenti trovati, da controllare a mano per stimare la precisione:
`tables/commenti_espliciti_campione.csv`.

## 3. Scenari (pipeline del 2.2c, iperparametri invariati)
{md(confronto)}

"utenti comuni": utenti di test presenti in tutti gli scenari ({len(comuni)}), per confrontare sugli stessi utenti.
"controllo": ogni utente perde lo stesso numero di commenti di "senza entrambi", scelti a caso (seme {SEED}): la
differenza tra "senza entrambi" e il controllo e' l'effetto delle scorciatoie, al netto della quantita' di testo.

## Per numero di commenti (utenti di test comuni, fascia dal numero originale di commenti)
{md(fasce.pivot(index='fascia', columns='scenario', values='macro-F1').reindex(FASCE + ['tutti']).reset_index()[['fascia'] + SCENARI])}

## 4. Parole con il peso maggiore (peso x dev. std della feature)
Nessuna rimozione:
{md(pesi['nessuna rimozione'])}

Senza entrambi:
{md(pesi['senza entrambi'])}

Grafico: `figures/shortcut_scenari.png`.
"""
    (RISULTATI / "report_s2_3b.md").write_text(testo, encoding="utf-8")
    log.info("Report: %s", RISULTATI / "report_s2_3b.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
