# ML4N-project

Project of the group I related to the course Machine Learning for Networking.
Progetto 7: *Patterns in comments from Reddit* (traccia in `docs/traccia_progetto7.pdf`).

## Struttura

```
ML4N-project/
├── README.md, requirements.txt, .gitignore
├── docs/
│   ├── traccia_progetto7.pdf        traccia del progetto
│   ├── scaletta_operazioni.docx     scaletta delle operazioni
│   ├── revisione_tecnica.pdf        revisione tecnica del 24 novembre
│   └── report_avanzamento.pdf       report di avanzamento (teoria, scelte, risultati)
└── codice/
    ├── run_all.sh                   esegue tutti gli script in ordine
    ├── comune.py                    parametri e pipeline condivisi dalla Sezione 2
    ├── preprocessing.py             funzioni di preprocessing del testo
    ├── s1_*.py, s2_*.py             uno script per punto della traccia (tabella sotto)
    ├── risultati_sezione1/
    │   ├── esplorazione/            1.1: report, tabelle, figure
    │   ├── preprocessing/           1.2: commenti preprocessati, riepilogo, esempi
    │   └── vettorizzazione/         1.3–1.4: BoW, TF-IDF, PCA
    └── risultati_sezione2/          2.1–2.3: report_s2_*.md, tabelle, figure, pipeline scelta
```

Il nome di ogni script dice il punto della traccia (`s2_2c_tuning.py` = Sezione 2, punto 2.2c) e il suo
report si chiama allo stesso modo (`report_s2_2c.md`).

| Script | Punto | Cosa fa |
| --- | --- | --- |
| `s1_1_esplorazione.py` | 1.1 | esplorazione, duplicati, valori mancanti, controlli sulle etichette |
| `s1_2_preprocessing.py` | 1.2 | pulizia e preprocessing di tutti i commenti (usa `preprocessing.py`) |
| `s1_3_bow.py` | 1.3a–b, 1.4a | aggregazione per utente, Bag of Words, soglie sul vocabolario |
| `s1_3_tfidf.py` | 1.3c–d | TF-IDF con la formula delle slide e confronto con la BoW |
| `s1_4_pca.py` | 1.4b | PCA, criteri per il numero di componenti, loading |
| `s2_1_dataset.py` | 2.1 | split stratificato 80/20, baseline, vettorizzazione sul solo training |
| `s2_2a_default.py` | 2.2a | Logistic Regression, Linear SVM e kNN con i parametri di default |
| `s2_2b_overfitting.py` | 2.2b | learning curve e validation curve |
| `s2_2c_tuning.py` | 2.2c | grid search con cross-validation, scelta della pipeline, test finale |
| `s2_2d_fasce.py` | 2.2d | prestazioni per numero di commenti, esperimento di troncamento |
| `s2_2_verifiche.py` | 2.2a, 2.2c | convergenza della Logistic Regression, tempi misurati senza concorrenza |
| `s2_3a_subreddit.py` | 2.3a | solo testo, solo subreddit, testo + subreddit |
| `s2_3b_shortcut.py` | 2.3b | rimozione di subreddit sbilanciati e commenti espliciti, scenario di controllo |

`comune.py` contiene i parametri condivisi (seme, quota di test, soglie `MIN_DF` e `MAX_DF`) e le pipeline:
tutti gli script della Sezione 2 li leggono da lì.

## Stato

| Sezione | Stato |
| --- | --- |
| 1. Esplorazione, preprocessing, vettorizzazione, PCA | completa |
| 2. Classificazione del genere (2.1–2.3) | completa |
| 3. Clustering dei commenti | da fare |
| 4. Sentiment | da fare |

## Come eseguire

1. Mettere in `codice/` i tre CSV del progetto: `data_supervised.csv`, `data_unsupervised.csv`,
   `target_supervised.csv` (non sono nel repository).
2. Creare l'ambiente e installare le librerie (sul cluster: `~/ml4n-env`, vedi la guida del cluster):

   ```bash
   python3 -m venv --system-site-packages ~/ml4n-env
   source ~/ml4n-env/bin/activate
   pip install -r requirements.txt
   ```

3. Eseguire tutto, dalla cartella `codice/`:

   ```bash
   nohup ./run_all.sh > log_run_all.txt 2>&1 &
   tail -f log_run_all.txt
   ```

   Il log riporta all'inizio le versioni delle librerie. Per registrare l'ambiente esatto:
   `pip freeze > requirements-cluster.txt`. Un singolo punto si riesegue con il suo script, per esempio
   `python3 s2_3b_shortcut.py`, purché i passi precedenti abbiano già prodotto i loro file.

Nel repository sono versionati i report, le tabelle e le figure; i file grandi (commenti preprocessati,
matrici, dataset puliti, split) si rigenerano con `run_all.sh`.

## Riproducibilità

- Tutti i passaggi casuali hanno un seme fisso: 42 per split, cross-validation e campionamenti, 15 per la PCA.
- `codice/risultati_sezione2/pipeline_migliore.joblib` è la pipeline scelta nel punto 2.2c, addestrata sui
  4.000 utenti di training. Si rigenera con `s2_2c_tuning.py` e va riletta con la stessa versione di
  scikit-learn con cui è stata salvata.
- Lo stemming usa `PorterStemmer` di nltk: con un'altra versione alcune radici possono cambiare.
