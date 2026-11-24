# ML4N Project 7 — Fase 3b (Sezione 2.2a): modelli con parametri di default

Training sui 4.000 utenti di training, valutazione su training e test (1.000 utenti). Baseline: macro-F1 0,422, accuracy 0,730.

## Sintesi
| modello | rappresentazione | accuracy training | accuracy test | macro-F1 training | macro-F1 test |
|---|---|---|---|---|---|
| Logistic Regression | BoW | 0.976 | 0.797 | 0.968 | 0.737 |
| Logistic Regression | TF-IDF | 0.997 | 0.796 | 0.996 | 0.74 |
| Linear SVM | BoW | 0.998 | 0.766 | 0.998 | 0.715 |
| Linear SVM | TF-IDF | 0.999 | 0.747 | 0.999 | 0.697 |
| kNN | BoW | 0.801 | 0.713 | 0.709 | 0.589 |
| kNN | TF-IDF | 0.799 | 0.73 | 0.704 | 0.608 |

Migliore sul test (macro-F1): **Logistic Regression con TF-IDF** — macro-F1 0.740, accuracy 0.796.

## Metriche complete
| modello | rappresentazione | insieme | accuracy | precision femminile | recall femminile | F1 maschile | F1 femminile | macro-F1 | tempo training [s] | convergenza |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | BoW | training | 0.9758 | 0.997 | 0.9129 | 0.9837 | 0.9531 | 0.9684 | 35.02 | NO (ConvergenceWarning) |
| Logistic Regression | BoW | test | 0.797 | 0.6324 | 0.5926 | 0.8626 | 0.6119 | 0.7372 | 35.02 | NO (ConvergenceWarning) |
| Logistic Regression | TF-IDF | training | 0.9972 | 1.0 | 0.9898 | 0.9981 | 0.9949 | 0.9965 | 36.04 | NO (ConvergenceWarning) |
| Logistic Regression | TF-IDF | test | 0.796 | 0.6241 | 0.6148 | 0.8607 | 0.6194 | 0.74 | 36.04 | NO (ConvergenceWarning) |
| Linear SVM | BoW | training | 0.9982 | 1.0 | 0.9935 | 0.9988 | 0.9967 | 0.9978 | 3.12 | si |
| Linear SVM | BoW | test | 0.766 | 0.5588 | 0.6333 | 0.8357 | 0.5938 | 0.7147 | 3.12 | si |
| Linear SVM | TF-IDF | training | 0.999 | 1.0 | 0.9963 | 0.9993 | 0.9981 | 0.9987 | 2.93 | si |
| Linear SVM | TF-IDF | test | 0.747 | 0.5262 | 0.6333 | 0.8199 | 0.5748 | 0.6974 | 2.93 | si |
| kNN | BoW | training | 0.8007 | 0.7092 | 0.443 | 0.8724 | 0.5454 | 0.7089 | 2.62 | si |
| kNN | BoW | test | 0.713 | 0.453 | 0.3037 | 0.8147 | 0.3636 | 0.5892 | 2.62 | si |
| kNN | TF-IDF | training | 0.799 | 0.7106 | 0.43 | 0.8717 | 0.5358 | 0.7038 | 2.85 | si |
| kNN | TF-IDF | test | 0.73 | 0.5 | 0.3185 | 0.8267 | 0.3891 | 0.6079 | 2.85 | si |

## Tempi di training e convergenza
| modello | rappresentazione | tempo training [s] | convergenza |
|---|---|---|---|
| Logistic Regression | BoW | 35.02 | NO (ConvergenceWarning) |
| Logistic Regression | TF-IDF | 36.04 | NO (ConvergenceWarning) |
| Linear SVM | BoW | 3.12 | si |
| Linear SVM | TF-IDF | 2.93 | si |
| kNN | BoW | 2.62 | si |
| kNN | TF-IDF | 2.85 | si |

Confusion matrix e classification report di ogni combinazione: `classification_reports_default.txt` e `figures/confusion_default_*.png`.
