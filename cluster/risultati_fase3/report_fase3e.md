# ML4N Project 7 — Fase 3e (Sezione 2.2d): prestazioni per numero di commenti

Pipeline valutata: quella scelta nel punto 2.2c (`pipeline_migliore.joblib`), addestrata sui 4.000 utenti
di training. Gruppi formati dai 1.000 utenti di test in base al numero di commenti.

## Fasce della traccia
| fascia | utenti | maschile | femminile | accuracy | macro-F1 | F1 femminile | recall femminile | macro-F1 baseline |
|---|---|---|---|---|---|---|---|---|
| 1-5 | 291 | 211 | 80 | 0.653 | 0.611 | 0.482 | 0.588 | 0.42 |
| 6-20 | 270 | 187 | 83 | 0.807 | 0.788 | 0.723 | 0.819 | 0.409 |
| 21-100 | 316 | 244 | 72 | 0.896 | 0.859 | 0.787 | 0.847 | 0.436 |
| >100 | 123 | 88 | 35 | 0.911 | 0.891 | 0.845 | 0.857 | 0.417 |
| tutti | 1000 | 730 | 270 | 0.803 | 0.767 | 0.677 | 0.763 | 0.422 |

## Fasce più fini
| fascia | utenti | maschile | femminile | accuracy | macro-F1 | F1 femminile | recall femminile | macro-F1 baseline |
|---|---|---|---|---|---|---|---|---|
| 1 | 90 | 71 | 19 | 0.644 | 0.533 | 0.304 | 0.368 | 0.441 |
| 2-3 | 116 | 86 | 30 | 0.638 | 0.599 | 0.475 | 0.633 | 0.426 |
| 4-5 | 85 | 54 | 31 | 0.682 | 0.671 | 0.609 | 0.677 | 0.388 |
| 6-10 | 124 | 88 | 36 | 0.79 | 0.766 | 0.69 | 0.806 | 0.415 |
| 11-20 | 146 | 99 | 47 | 0.822 | 0.806 | 0.75 | 0.83 | 0.404 |
| 21-50 | 187 | 142 | 45 | 0.877 | 0.84 | 0.763 | 0.822 | 0.432 |
| 51-100 | 129 | 102 | 27 | 0.922 | 0.889 | 0.828 | 0.889 | 0.442 |
| 101-300 | 95 | 68 | 27 | 0.905 | 0.887 | 0.842 | 0.889 | 0.417 |
| >300 | 28 | 20 | 8 | 0.929 | 0.905 | 0.857 | 0.75 | 0.417 |
| tutti | 1000 | 730 | 270 | 0.803 | 0.767 | 0.677 | 0.763 | 0.422 |

## Dove cadono gli errori
| fascia | utenti sbagliati | quota sbagliata | quota sul totale degli errori |
|---|---|---|---|
| 1-5 | 101 | 0.347 | 0.513 |
| 6-20 | 52 | 0.193 | 0.264 |
| 21-100 | 33 | 0.104 | 0.168 |
| >100 | 11 | 0.089 | 0.056 |

Confusion matrix per fascia (VN, FP, FN, VP): `tables/prestazioni_per_fascia.csv`. Grafico: `figures/prestazioni_per_fascia.png`.

Attenzione: nelle fasce con pochi utenti (in particolare la classe femminile) le metriche variano molto
cambiando anche pochi utenti; vanno lette insieme al numero di utenti indicato. Le fasce contengono anche
utenti diversi: il confronto tra fasce da solo non dimostra che la causa sia il numero di commenti.

## Esperimento di troncamento (stessi utenti, meno commenti)
123 utenti di test con più di 100 commenti (35 femminili); per ogni k, 10 sorteggi
di k commenti per utente. La pipeline non viene riaddestrata.

| commenti tenuti | macro-F1 media | dev. std tra i sorteggi |
|---|---|---|
| 1 | 0.616 | 0.041 |
| 2 | 0.664 | 0.035 |
| 5 | 0.715 | 0.029 |
| 10 | 0.783 | 0.023 |
| 20 | 0.835 | 0.035 |
| 50 | 0.86 | 0.017 |
| 100 | 0.868 | 0.01 |
| tutti | 0.891 | 0.0 |

Se la macro-F1 degli stessi utenti scende quando si tolgono commenti, l'aumento osservato tra le fasce è
dovuto (almeno in parte) alla quantità di testo e non solo al tipo di utente. Grafico: `figures/troncamento_commenti.png`.
