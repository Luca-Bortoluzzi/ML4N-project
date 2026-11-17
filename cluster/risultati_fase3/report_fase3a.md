# ML4N Project 7 — Fase 3a (Sezione 2.1): preparazione del dataset

## Dati
- Utenti: 5,000; autori duplicati: 0; etichette non valide: 0.
- Utenti con documento vuoto dopo il preprocessing: 8.
- Feature usata: solo il testo aggregato per utente (`doc`). `author` è usato solo come chiave, come richiesto dalla traccia.

## Split stratificato (80% / 20%, random_state = 42)
| insieme | utenti | maschile (0) | femminile (1) | quota femminile [%] | commenti (mediana) | token (mediana) | documenti vuoti |
|---|---|---|---|---|---|---|---|
| training | 4000 | 2921 | 1079 | 26.98 | 16.0 | 194.0 | 6 |
| test | 1000 | 730 | 270 | 27.0 | 16.0 | 197.0 | 2 |

Utenti per fascia di commenti (servirà al punto 2.2d):

| fascia_commenti | training - maschile | training - femminile | test - maschile | test - femminile | test - totale |
|---|---|---|---|---|---|
| 1-5 | 767 | 348 | 211 | 80 | 291 |
| 6-20 | 786 | 298 | 187 | 83 | 270 |
| 21-100 | 922 | 286 | 244 | 72 | 316 |
| >100 | 446 | 147 | 88 | 35 | 123 |

## Baseline: predice sempre la classe maggioritaria
| insieme | accuracy | F1 maschile | F1 femminile | macro-F1 |
|---|---|---|---|---|
| training | 0.7302 | 0.8441 | 0.0 | 0.422 |
| test | 0.73 | 0.8439 | 0.0 | 0.422 |

## Vettorizzazione addestrata solo sul training (MIN_DF = 10, MAX_DF = 0.7)
| misura | valore |
|---|---|
| parole nel vocabolario (fit sul training) | 12,272 |
| parole nel vocabolario se fit su tutti i 5.000 utenti | 13,509 |
| parole in piu' con il fit su tutti (scelte anche grazie al test: leakage evitato) | 1,237 |
| parole del vocabolario di training assenti nel fit su tutti | 0 |
| token del test esclusi perche' troppo comuni (just, like) [%] | 2.5 |
| token del test fuori vocabolario perche' rari o mai visti nel training [%] | 5.0 |
| matrice di training (utenti x parole) | 4,000 x 12,272 |
| matrice di test (utenti x parole) | 1,000 x 12,272 |
| sparsita' training / test [%] | 97.34 / 97.64 |
| utenti con vettore vuoto training / test | 14 / 2 |
| idf calcolato su (utenti di training) | 4,000 |
| idf min / max | 0.48 / 5.99 |
