# ML4N-project

Project of the group I related to the course Machine Learning for Networking.
Progetto 7: *Patterns in comments from Reddit* (traccia in `7_Reddit/ML4N_project7_reddit.pdf`).

## Contenuto

| Percorso | Cosa contiene |
| --- | --- |
| `7_Reddit/` | traccia del progetto, scaletta delle operazioni, report di avanzamento |
| `cluster/` | script Python, uno per passo della traccia, e i loro risultati |
| `requirements.txt` | librerie Python necessarie |

Script in `cluster/`, nell'ordine in cui `run_all.sh` li esegue:

| Script | Punto della traccia | Cosa fa |
| --- | --- | --- |
| `ML4N_phase1.py` | 1.1 | esplorazione, duplicati, valori mancanti, controlli sulle etichette |
| `fase2_clean.py` (+ `preprocessing.py`) | 1.2 | pulizia e preprocessing di tutti i commenti |
| `fase2a_bow.py` | 1.3a–b, 1.4a | aggregazione per utente, Bag of Words, soglie sul vocabolario |
| `fase2b_tfidf.py` | 1.3c–d | TF-IDF con la formula delle slide e confronto con la BoW |
| `fase2c_pca.py` | 1.4b | PCA, criteri per il numero di componenti, loading |
| `fase3a_dataset.py` | 2.1 | split stratificato 80/20, baseline, vettorizzazione sul solo training |
| `fase3b_modelli_default.py` | 2.2a | Logistic Regression, Linear SVM e kNN con i parametri di default |
| `fase3c_overfitting.py` | 2.2b | learning curve e validation curve |
| `fase3d_tuning.py` | 2.2c | grid search con cross-validation, scelta della pipeline, test finale |
| `fase3e_fasce.py` | 2.2d | prestazioni per numero di commenti, esperimento di troncamento |
| `fase3f_verifiche.py` | 2.2a, 2.2c | convergenza della Logistic Regression, tempi misurati senza concorrenza |

`supervised_common.py` contiene i parametri condivisi (seme, quota di test, soglie `MIN_DF` e `MAX_DF`) e le
pipeline: tutti gli script li leggono da lì.

## Come eseguire

1. Mettere in `cluster/` i tre CSV del progetto: `data_supervised.csv`, `data_unsupervised.csv`,
   `target_supervised.csv` (non sono nel repository).
2. Creare l'ambiente e installare le librerie (sul cluster: `~/ml4n-env`, vedi la guida del cluster):

   ```bash
   python3 -m venv --system-site-packages ~/ml4n-env
   source ~/ml4n-env/bin/activate
   pip install -r requirements.txt
   ```

3. Eseguire tutto, dalla cartella `cluster/`:

   ```bash
   nohup ./run_all.sh > log_run_all.txt 2>&1 &
   tail -f log_run_all.txt
   ```

   Il log riporta all'inizio le versioni delle librerie. Per registrare l'ambiente esatto:
   `pip freeze > requirements-cluster.txt`.

Ogni script scrive tabelle, figure e un report markdown in `cluster/risultati_fase1/`, `risultati_fase2/` e
`risultati_fase3/`. Nel repository sono versionati i report, le tabelle e le figure; i file grandi (commenti
preprocessati, matrici, dataset puliti) si rigenerano con `run_all.sh`.

## Riproducibilità

- Tutti i passaggi casuali hanno un seme fisso: 42 per split e cross-validation, 15 per la PCA.
- `cluster/risultati_fase3/pipeline_migliore.joblib` è la pipeline scelta nel punto 2.2c, addestrata sui
  4.000 utenti di training. Si rigenera con `fase3d_tuning.py` e va riletta con la stessa versione di
  scikit-learn con cui è stata salvata.
- Lo stemming usa `PorterStemmer` di nltk: con un'altra versione alcune radici possono cambiare.
