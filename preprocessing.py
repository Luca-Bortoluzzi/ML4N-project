# -*- coding: utf-8 -*-
"""ML4N Project 7 — Fase 2: funzioni di preprocessing del testo.

Implementa la sequenza della slide 74 di "3_ML4N_DataPreProcessing":
    1. Document splitting   -> NON applicato: i commenti sono testi brevi
                               ("social posts"), 1 commento = 1 documento.
    2. Tokenizzazione
    3. Case normalization   (minuscolo)
    4. Stopword removal     (lista inglese di scikit-learn)
    5. Stemming             (PorterStemmer di nltk: sklearn non fa stemming)

Prima dei 5 passi si applica una rimozione del "rumore" specifico di Reddit
(URL, entita' HTML, righe citate, link markdown), richiesta dal punto 1.2a
della traccia ("clean and uniform the text").

Questo modulo contiene solo funzioni: viene importato da fase2_clean.py e
dalle fasi successive, cosi' la pulizia e' identica ovunque.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ----------------------------------------------------------------------
# Parametri della pulizia (documentarli nella relazione)
# ----------------------------------------------------------------------
LUNGHEZZA_MINIMA_TOKEN = 2          # scarta i token di 1 lettera ("s", "t", "u")
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)

# Commenti senza contenuto (Reddit li sostituisce con questi segnaposto)
SEGNAPOSTO_RIMOSSI = frozenset({"[deleted]", "[removed]"})

# ----------------------------------------------------------------------
# Espressioni regolari (compilate una volta sola)
# ----------------------------------------------------------------------
_RE_RIGA_CITATA = re.compile(r"^\s*>.*$", flags=re.MULTILINE)       # "> testo di un altro utente"
_RE_LINK_MARKDOWN = re.compile(r"\[([^\]]*)\]\([^)]*\)")            # [testo](url) -> testo
_RE_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_RE_MENZIONE_UTENTE = re.compile(r"(?<!\w)/?u/[A-Za-z0-9_-]+")      # u/nome, /u/nome
_RE_APOSTROFI = re.compile(r"[’‘`´]")                              # apostrofi tipografici -> '
_RE_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*")                       # parole alfabetiche, anche "don't"

_STEMMER = PorterStemmer()


def remove_noise(text: str) -> str:
    """Rimozione del rumore specifico di Reddit (traccia, punto 1.2a)."""
    text = html.unescape(text)                  # &amp; -> &, &gt; -> >
    text = _RE_RIGA_CITATA.sub(" ", text)       # le citazioni sono parole di ALTRI utenti
    text = _RE_LINK_MARKDOWN.sub(r"\1", text)   # tiene il testo del link, toglie l'URL
    text = _RE_URL.sub(" ", text)
    text = _RE_MENZIONE_UTENTE.sub(" ", text)   # i nomi utente non sono contenuto
    text = _RE_APOSTROFI.sub("'", text)
    return text


def tokenize(text: str) -> list[str]:
    """Tokenizzazione + case normalization (slide 76-78).

    Tiene solo sequenze alfabetiche: punteggiatura, numeri, emoji e simboli
    vengono scartati. Le forme con apostrofo ("don't") restano un token unico,
    cosi' possono essere confrontate con la lista di stopword.
    """
    return _RE_TOKEN.findall(text.lower())


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Stopword removal (slide 79) con la lista inglese di scikit-learn.

    L'apostrofo viene tolto dopo il confronto con la lista ("don't" -> "dont"),
    poi si scartano i token troppo corti.
    """
    out = []
    for tok in tokens:
        if tok in STOP_WORDS:
            continue
        tok = tok.replace("'", "")
        if len(tok) >= LUNGHEZZA_MINIMA_TOKEN and tok not in STOP_WORDS:
            out.append(tok)
    return out


@lru_cache(maxsize=500_000)
def _stem(token: str) -> str:
    # Cache: le stesse parole si ripetono milioni di volte, cosi' ognuna
    # viene ridotta alla radice una volta sola.
    return _STEMMER.stem(token)


def stem(tokens: list[str]) -> list[str]:
    """Stemming (slide 74) con PorterStemmer."""
    return [_stem(tok) for tok in tokens]


def preprocess(text: object) -> str:
    """Applica in sequenza: rumore -> token/minuscolo -> stopword -> stemming.

    Restituisce i token uniti da spazi (formato pronto per CountVectorizer).
    Un testo mancante o un segnaposto [deleted]/[removed] diventa stringa vuota.
    """
    if not isinstance(text, str) or text.strip().lower() in SEGNAPOSTO_RIMOSSI:
        return ""
    return " ".join(stem(remove_stopwords(tokenize(remove_noise(text)))))


def preprocess_many(texts: list[object]) -> list[str]:
    """Versione su lista, usata per l'esecuzione parallela a blocchi."""
    return [preprocess(t) for t in texts]
