# ML4N Project 7 — Sezione 2.3a: feature dei subreddit

Feature: per ogni utente, la frazione dei suoi commenti scritta in ciascun subreddit. Subreddit tenuti: quelli
con almeno `min_utenti` utenti di training (scelto in CV). Subreddit distinti nel training: 3,329; colonne
del modello "solo subreddit": 2,525. Utenti di test senza nessun subreddit tenuto: 7.

Modello: Linear SVM con class_weight = balanced, come la pipeline scelta nel punto 2.2c. Per "solo subreddit"
e "testo + subreddit" C e min_utenti (e per "solo subreddit" anche la standardizzazione) sono scelti con CV a 5
fold sul training (macro-F1, configurazione con la validazione migliore). In "testo + subreddit" la parte di testo
usa gli iperparametri del 2.2c e tutte le colonne sono standardizzate (StandardScaler, with_mean=False), perche'
TF-IDF e frazioni hanno scale diverse. Il test e' usato una sola volta per pipeline.

## Confronto
| feature | iperparametri | macro-F1 CV | dev. std CV | macro-F1 training | macro-F1 test | accuracy test | F1 femminile test | tempo training [s] | iterazioni | converge |
|---|---|---|---|---|---|---|---|---|---|---|
| solo testo | come nel 2.2c | 0.771 | 0.0099 | 0.8427 | 0.7674 | 0.803 | 0.6765 | 2.95 | 20 | si' |
| solo subreddit | min_utenti = 2, C = 1, non standardizzate | 0.7679 | 0.0125 | 0.8584 | 0.7581 | 0.802 | 0.6551 | 0.4 | 6 | si' |
| testo + subreddit | min_utenti = 10, C = 1e-05 | 0.7912 | 0.0063 | 0.891 | 0.7865 | 0.82 | 0.702 | 3.34 | 12 | si' |

## Per numero di commenti (test)
| fascia | utenti | femminile | macro-F1 solo testo | macro-F1 solo subreddit | macro-F1 testo + subreddit |
|---|---|---|---|---|---|
| 1-5 | 291 | 80 | 0.611 | 0.666 | 0.652 |
| 6-20 | 270 | 83 | 0.788 | 0.76 | 0.79 |
| 21-100 | 316 | 72 | 0.859 | 0.821 | 0.864 |
| >100 | 123 | 35 | 0.891 | 0.823 | 0.92 |
| tutti | 1000 | 270 | 0.767 | 0.758 | 0.787 |

## Subreddit con il peso maggiore (modello "solo subreddit")
| verso | subreddit | peso |
|---|---|---|
| femminile | TrollXChromosomes | 3.613 |
| femminile | MakeupAddiction | 3.044 |
| femminile | TheGirlSurvivalGuide | 3.006 |
| femminile | TwoXChromosomes | 2.873 |
| femminile | femalefashionadvice | 2.756 |
| femminile | Parenting | 2.507 |
| femminile | GirlGamers | 2.397 |
| femminile | actuallesbians | 2.393 |
| femminile | piercing | 2.267 |
| femminile | LadyBoners | 2.214 |
| femminile | childfree | 2.14 |
| femminile | RedditLaqueristas | 2.137 |
| femminile | AskWomen | 2.089 |
| femminile | creepyPMs | 2.052 |
| femminile | BabyBumps | 1.964 |
| maschile | nfl | -2.729 |
| maschile | technology | -2.441 |
| maschile | malefashionadvice | -2.172 |
| maschile | Android | -1.835 |
| maschile | 4chan | -1.78 |
| maschile | nba | -1.731 |
| maschile | cars | -1.703 |
| maschile | magicTCG | -1.685 |
| maschile | leagueoflegends | -1.649 |
| maschile | pcmasterrace | -1.631 |
| maschile | movies | -1.541 |
| maschile | DotA2 | -1.49 |
| maschile | dayz | -1.468 |
| maschile | SquaredCircle | -1.415 |
| maschile | sweden | -1.408 |

Grafico: `figures/subreddit_confronto.png`. Griglie complete: `tables/subreddit_cv_*.csv`.
