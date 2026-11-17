# ML4N Project 7 — Fase 3c (Sezione 2.2b): overfitting e underfitting

## 1. Divario tra training e test (parametri di default)
| modello | rappresentazione | training | test | divario (training - test) |
|---|---|---|---|---|
| Logistic Regression | BoW | 0.968 | 0.737 | 0.231 |
| Logistic Regression | TF-IDF | 0.996 | 0.74 | 0.257 |
| Linear SVM | BoW | 0.998 | 0.715 | 0.283 |
| Linear SVM | TF-IDF | 0.999 | 0.697 | 0.301 |
| kNN | BoW | 0.709 | 0.589 | 0.12 |
| kNN | TF-IDF | 0.704 | 0.608 | 0.096 |

## 2. Learning curve (CV a 5 fold sui 4.000 utenti di training)
| modello | utenti (min) | macro-F1 val. (min) | utenti (max) | macro-F1 val. (max) | macro-F1 training (max) |
|---|---|---|---|---|---|
| Logistic Regression | 320 | 0.6414 | 3200 | 0.7164 | 0.9967 |
| Linear SVM | 320 | 0.602 | 3200 | 0.6772 | 0.998 |
| kNN | 320 | 0.5437 | 3200 | 0.5725 | 0.7 |

Tabella completa: `tables/learning_curve.csv`; grafico: `figures/learning_curve.png`.

## 3. Validation curve
| modello | parametro | valore | macro-F1 training | macro-F1 validazione | dev. std validazione | divario |
|---|---|---|---|---|---|---|
| Logistic Regression | C | 0.0001 | 0.6311 | 0.5708 | 0.0091 | 0.0604 |
| Logistic Regression | C | 0.001 | 0.7974 | 0.6669 | 0.0136 | 0.1305 |
| Logistic Regression | C | 0.01 | 0.921 | 0.7294 | 0.0075 | 0.1916 |
| Logistic Regression | C | 0.1 | 0.9837 | 0.7295 | 0.022 | 0.2542 |
| Logistic Regression | C | 1.0 | 0.9967 | 0.717 | 0.0244 | 0.2796 |
| Logistic Regression | C | 10.0 | 0.9983 | 0.7233 | 0.0173 | 0.2751 |
| Logistic Regression | C | 100.0 | 0.9988 | 0.739 | 0.0185 | 0.2598 |
| Linear SVM | C | 9.999999999999999e-06 | 0.6816 | 0.6494 | 0.0236 | 0.0323 |
| Linear SVM | C | 0.0001 | 0.7333 | 0.6632 | 0.0176 | 0.0701 |
| Linear SVM | C | 0.001 | 0.8224 | 0.6945 | 0.0174 | 0.1279 |
| Linear SVM | C | 0.01 | 0.9247 | 0.7111 | 0.0077 | 0.2137 |
| Linear SVM | C | 0.1 | 0.9846 | 0.7013 | 0.0171 | 0.2833 |
| Linear SVM | C | 1.0 | 0.998 | 0.6772 | 0.0181 | 0.3208 |
| Linear SVM | C | 10.0 | 0.9992 | 0.6732 | 0.0179 | 0.326 |
| kNN | n_neighbors | 201.0 | 0.4648 | 0.4644 | 0.0188 | 0.0004 |
| kNN | n_neighbors | 101.0 | 0.5043 | 0.5038 | 0.0179 | 0.0005 |
| kNN | n_neighbors | 51.0 | 0.5308 | 0.5218 | 0.0147 | 0.009 |
| kNN | n_neighbors | 21.0 | 0.5681 | 0.5475 | 0.0035 | 0.0206 |
| kNN | n_neighbors | 11.0 | 0.6134 | 0.5678 | 0.0099 | 0.0456 |
| kNN | n_neighbors | 5.0 | 0.7006 | 0.572 | 0.0091 | 0.1285 |
| kNN | n_neighbors | 3.0 | 0.7744 | 0.5697 | 0.0038 | 0.2048 |
| kNN | n_neighbors | 1.0 | 0.9981 | 0.5692 | 0.0123 | 0.4289 |

Grafico: `figures/validation_curve.png`.
