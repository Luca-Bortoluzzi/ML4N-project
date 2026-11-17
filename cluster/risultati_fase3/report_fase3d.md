# ML4N Project 7 — Fase 3d (Sezione 2.2c): tuning con cross-validation

Griglia: completa. CV stratificata a 5 fold sui 4.000 utenti di training, metrica macro-F1.

## Migliore configurazione di ogni modello (validazione)
| modello | rappresentazione | max_df | min_df | C | class_weight | macro-F1 validazione | dev. std validazione | macro-F1 training | accuracy validazione | tempo training [s] | metric | n_neighbors | weights |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | TF-IDF | 0.7 | 5 | 0.0001 | balanced | 0.7782 | 0.0131 | 0.8694 | 0.8317 | 2.68 | — | — | — |
| Linear SVM | TF-IDF | 0.7 | 10 | 1e-05 | balanced | 0.7767 | 0.0103 | 0.8541 | 0.8125 | 2.46 | — | — | — |
| kNN | TF-IDF | 0.7 | 5 | — | — | 0.7462 | 0.0073 | 0.7611 | 0.8108 | 2.4 | cosine | 51 | uniform |

## Scelta
Regola: macro-F1 di validazione massima; tra le configurazioni entro una deviazione standard dalla migliore,
la piu' veloce da addestrare (vedi `tables/tuning_candidate_entro_1std.csv`).

**Pipeline scelta: Linear SVM con TF-IDF** — macro-F1 CV 0.7710 ± 0.0099, tempo di training 2.38 s.
Parametri: `{'bow__max_df': 0.5, 'bow__min_df': 20, 'modello__C': 1e-05, 'modello__class_weight': 'balanced', 'tfidf': 'TF-IDF'}`

## Valutazione finale sul test (una sola volta)
| pipeline | macro-F1 training | macro-F1 test | accuracy test | F1 femminile test | tempo training [s] |
|---|---|---|---|---|---|
| Linear SVM (TF-IDF) — SCELTA | 0.8427 | 0.7674 | 0.803 | 0.6765 | 2.85 |
| Logistic Regression (TF-IDF) — informativo | 0.867 | 0.8041 | 0.85 | 0.7093 | 5.26 |
| kNN (TF-IDF) — informativo | 0.7612 | 0.7743 | 0.82 | 0.6727 | 2.86 |

Le righe "informativo" non sono state usate per la scelta. Con i parametri di default la stessa combinazione aveva macro-F1 test 0.697 (punto 2.2a).

Confusion matrix della pipeline scelta (test):

|  | pred. maschile | pred. femminile |
|---|---|---|
| vero maschile | 597 | 133 |
| vero femminile | 64 | 206 |

```
              precision    recall  f1-score   support

    maschile      0.903     0.818     0.858       730
   femminile      0.608     0.763     0.677       270

    accuracy                          0.803      1000
   macro avg      0.755     0.790     0.767      1000
weighted avg      0.823     0.803     0.809      1000

```
