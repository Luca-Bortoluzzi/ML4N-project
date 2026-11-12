# ML4N Project 7 — Fase 1: esplorazione e qualità dei dati

## Ambito

Il report copre le analisi esplorative (EDA), la verifica delle etichette, la gestione delle righe duplicate
e la preparazione dei CSV. Il testo dei commenti viene conservato nella forma originale:
tokenizzazione, stopword removal, lemmatizzazione, BoW, TF-IDF e PCA/SVD saranno svolti in una fase successiva.

## Riepilogo dataset

| Dataset | Righe originali | Duplicati esatti | Altre righe rimosse | Righe pulite | Autori | Subreddit |
|---|---:|---:|---:|---:|---:|---:|
| supervised | 296.042 | 7 | 0 | 296.035 | 5.000 | 3.468 |
| unsupervised | 1.107.946 | 256.137 | 8 | 851.801 | 14.999 | 3.970 |

## Indicatori descrittivi sui dati puliti

### Supervised

- Lunghezza media del testo: 183,16 caratteri.
- Lunghezza mediana: 90,00 caratteri.
- Commenti medi per autore: 59,21.
- Commenti mediani per autore: 16,00.
- Intervallo dei dati (UTC): 2014-01-01T00:00:04+00:00 — 2014-01-31T23:59:52+00:00.
- Timestamp non validi: 0.
- Commenti '[deleted]': 1; '[removed]': 0.
- Quota dei commenti nei subreddit principali: 40,84%.

### Unsupervised

- Lunghezza media del testo: 183,02 caratteri.
- Lunghezza mediana: 92,00 caratteri.
- Commenti medi per autore: 56,79.
- Commenti mediani per autore: 16,00.
- Intervallo dei dati (UTC): 2014-01-01T00:00:02+00:00 — 2014-01-31T23:59:49+00:00.
- Timestamp non validi: 0.
- Commenti '[deleted]': 7; '[removed]': 1.
- Quota dei commenti nei subreddit principali: 39,44%.

## Classi supervisionate

- Utenti etichettati: 5.000.
- Classe 0 (maschile): 3.651 (73,02%).
- Classe 1 (femminile): 1.349 (26,98%).
- Le etichette sono auto-dichiarate e non costituiscono una verifica indipendente dell'identità di genere.

## Controlli di integrità

| Controllo | Valore | Esito |
|---|---:|---|
| Etichette non valide | 0 | OK |
| Autori duplicati nel target | 0 | OK |
| Autori con etichette conflittuali | 0 | OK |
| Autori supervised senza target | 0 | OK |
| Target senza commenti supervised | 0 | OK |
| Autori in entrambi i dataset | 0 | OK |

## Interpretazione e cautele

- I duplicati esatti sono eliminati; i testi identici con metadati differenti non sono automaticamente cancellati.
- La distribuzione dei commenti per autore può essere molto sbilanciata: non confondere percentuali per commento e per utente.
- Negli istogrammi della lunghezza il 99° percentile è un limite **grafico**, mentre le ECDF includono tutti i dati.
- I grafici temporali usano UTC, non l'ora locale.
- `[deleted]` e `[removed]` sono segnalati ma mantenuti: decidere e documentare una politica nella fase successiva.
- Il testo rimane originale anche per consentire una corretta analisi VADER nella fase Sentiment Analysis.
- L'EDA non stabilisce una soglia ottimale per modelli ML e non produce ancora vettori numerici.

## File generati

- `tables/`: statistiche, frequenze, distribuzioni, controllo etichette e integrità.
- `figures/`: istogrammi, ECDF, subreddit, grafici temporali e confronto tra classi.
- `data_clean/`: dati ripuliti dai duplicati esatti (se l'esportazione è abilitata).
