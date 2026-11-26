# ML4N Project 7 — Sezione 2.3b: rimozione delle scorciatoie

## 1. Subreddit sbilanciati (stimati sui soli utenti di training)
Regola di Bayes con smoothing di Laplace: un subreddit e' sbilanciato verso F se, da solo, porta la probabilita'
a posteriori P(F|s) sopra 1/2, cioe' LR = P(s|F)/P(s|M) > P(M)/P(F) = 2.707; verso M se LR < 0.369.
Solo subreddit con almeno 10 utenti di training.
Subreddit sbilanciati: 276 (65 verso F,
211 verso M); i loro commenti sono il 30.9% del totale.

Verso F (primi 15):
| subreddit | verso | utenti | femminili | LR | P(F|s) |
|---|---|---|---|---|---|
| TheGirlSurvivalGuide | femminile | 35 | 34 | 47.3196 | 0.9459 |
| RedditLaqueristas | femminile | 33 | 32 | 44.6156 | 0.9428 |
| ABraThatFits | femminile | 15 | 15 | 43.2636 | 0.9411 |
| weddingplanning | femminile | 14 | 14 | 40.5597 | 0.9374 |
| makeupexchange | femminile | 14 | 14 | 40.5597 | 0.9374 |
| RandomActsofMakeup | femminile | 14 | 14 | 40.5597 | 0.9374 |
| MakeupAddiction | femminile | 156 | 146 | 36.135 | 0.9303 |
| femalefashionadvice | femminile | 41 | 39 | 36.053 | 0.9302 |
| xxketo | femminile | 12 | 12 | 35.1517 | 0.9285 |
| MakeupAddicts | femminile | 12 | 12 | 35.1517 | 0.9285 |
| curlyhair | femminile | 11 | 11 | 32.4477 | 0.923 |
| BeautyDiagrams | femminile | 11 | 11 | 32.4477 | 0.923 |
| FancyFollicles | femminile | 35 | 33 | 30.6451 | 0.9188 |
| xxfitness | femminile | 34 | 32 | 29.7438 | 0.9166 |
| BabyBumps | femminile | 22 | 20 | 18.9278 | 0.8749 |

Verso M (primi 15):
| subreddit | verso | utenti | femminili | LR | P(F|s) |
|---|---|---|---|---|---|
| dayz | maschile | 57 | 0 | 0.0466 | 0.0169 |
| cars | maschile | 52 | 0 | 0.051 | 0.0185 |
| guns | maschile | 39 | 0 | 0.0676 | 0.0244 |
| malefashionadvice | maschile | 117 | 2 | 0.0699 | 0.0252 |
| NoFap | maschile | 37 | 0 | 0.0712 | 0.0256 |
| frugalmalefashion | maschile | 36 | 0 | 0.0731 | 0.0263 |
| battlefield_4 | maschile | 32 | 0 | 0.0819 | 0.0294 |
| space | maschile | 31 | 0 | 0.0845 | 0.0303 |
| Autos | maschile | 31 | 0 | 0.0845 | 0.0303 |
| seduction | maschile | 30 | 0 | 0.0872 | 0.0312 |
| bodybuilding | maschile | 30 | 0 | 0.0872 | 0.0312 |
| nfl | maschile | 217 | 6 | 0.0893 | 0.0319 |
| Guitar | maschile | 29 | 0 | 0.0901 | 0.0322 |
| KerbalSpaceProgram | maschile | 28 | 0 | 0.0932 | 0.0333 |
| pcgaming | maschile | 27 | 0 | 0.0966 | 0.0344 |

Elenco completo: `tables/subreddit_sbilanciati.csv`.

## 2. Commenti in cui l'autore rivela il genere
Espressioni regolari sul testo originale (maiuscole e minuscole indifferenti):
- **dichiarazione**: `\b(?:i\s*'?\s*m|i\s+am)\s+an?\s+(?:(?:straight|gay|single|young|old|married|tall|short|black|white|bi|trans|grown|middle-aged)\s+)?(?:man|woman|guy|girl|dude|lady|gal|male|female|mom|mum|mother|dad|father|wife|husband|boyfriend|girlfriend)\b|\bi\s*'?\s*m\s+(?:fe)?male\b|\bi\s+am\s+(?:fe)?male\b|\bas\s+an?\s+(?:(?:straight|gay|single|young|old|married|tall|short|black|white|bi|trans|grown|middle-aged)\s+)?(?:man|woman|guy|girl|dude|lady|gal|male|female|mom|mum|mother|dad|father|wife|husband|boyfriend|girlfriend)(?:\s*,\s*(?:i|i'm|i've|i'd|im|my|me|we)\b|\s+(?:who|myself|i|am|here)\b|\s*\.\.\.)`
- **partner**: `\bmy\s+(?:husband|wife|hubby|boyfriend|girlfriend|gf|fianc[eé]e?|bf(?!\s+(?:at|around|of)\b)(?!\s*~?\d))\b`
- **tag eta-genere**: `(?<![-\d])[\(\[]\s*(?:1[3-9]|[2-6]\d)\s*/?\s*[mf]\s*[\)\]]|[\(\[]\s*[mf]\s*/?\s*(?:1[3-9]|[2-6]\d)\s*[\)\]]`

| regola | commenti | quota commenti | utenti |
|---|---|---|---|
| dichiarazione | 491 | 0.0017 | 346 |
| partner | 1880 | 0.0064 | 658 |
| tag eta-genere | 8 | 0.0 | 8 |
| almeno una regola | 2362 | 0.008 | 871 |

Campione casuale di 60 commenti trovati, da controllare a mano per stimare la precisione:
`tables/commenti_espliciti_campione.csv`.

## 3. Scenari (pipeline del 2.2c, iperparametri invariati)
| scenario | commenti tolti | quota commenti tolti | utenti training | utenti test | macro-F1 CV | dev. std CV | macro-F1 test | accuracy test | F1 femminile test | tempo training [s] | iterazioni | converge | differenza test | macro-F1 test (utenti comuni) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nessuna rimozione | 0 | 0.0 | 4000 | 1000 | 0.771 | 0.0099 | 0.7674 | 0.803 | 0.6765 | 2.95 | 21 | si' | 0.0 | 0.769 |
| senza subreddit sbilanciati | 91568 | 0.3093 | 3795 | 948 | 0.7059 | 0.0118 | 0.7182 | 0.7532 | 0.6189 | 2.0 | 22 | si' | -0.0492 | 0.718 |
| senza commenti espliciti | 2362 | 0.008 | 3996 | 999 | 0.7611 | 0.0163 | 0.7525 | 0.7878 | 0.6592 | 2.86 | 21 | si' | -0.0149 | 0.752 |
| senza entrambi | 93103 | 0.3145 | 3788 | 948 | 0.6936 | 0.0128 | 0.7047 | 0.7405 | 0.6019 | 2.02 | 20 | si' | -0.0627 | 0.705 |
| controllo: stessi commenti tolti a caso | 93103 | 0.3145 | 3788 | 948 | 0.7314 | 0.0128 | 0.7562 | 0.789 | 0.6667 | 2.03 | 21 | si' | -0.0112 | 0.756 |

"utenti comuni": utenti di test presenti in tutti gli scenari (948), per confrontare sugli stessi utenti.
"controllo": ogni utente perde lo stesso numero di commenti di "senza entrambi", scelti a caso (seme 42): la
differenza tra "senza entrambi" e il controllo e' l'effetto delle scorciatoie, al netto della quantita' di testo.

## Per numero di commenti (utenti di test comuni, fascia dal numero originale di commenti)
| fascia | nessuna rimozione | senza subreddit sbilanciati | senza commenti espliciti | senza entrambi | controllo: stessi commenti tolti a caso |
|---|---|---|---|---|---|
| 1-5 | 0.593 | 0.541 | 0.561 | 0.539 | 0.599 |
| 6-20 | 0.787 | 0.721 | 0.778 | 0.712 | 0.754 |
| 21-100 | 0.859 | 0.805 | 0.842 | 0.773 | 0.842 |
| >100 | 0.891 | 0.873 | 0.882 | 0.873 | 0.882 |
| tutti | 0.769 | 0.718 | 0.752 | 0.705 | 0.756 |

## 4. Parole con il peso maggiore (peso x dev. std della feature)
Nessuna rimozione:
| verso femminile | peso F | quota 1 utente F | verso maschile | peso M | quota 1 utente M |
|---|---|---|---|---|---|
| love | 0.0472 | 0.06 | jon | -0.1768 | 0.92 |
| femal | 0.0463 | 0.09 | fuck | -0.0593 | 0.05 |
| okay | 0.046 | 0.15 | team | -0.0451 | 0.04 |
| smile | 0.0438 | 0.43 | car | -0.0451 | 0.1 |
| babi | 0.0421 | 0.12 | game | -0.0342 | 0.07 |
| makeup | 0.0376 | 0.12 | man | -0.0341 | 0.04 |
| feel | 0.0365 | 0.02 | doge | -0.0309 | 0.25 |
| husband | 0.0355 | 0.07 | play | -0.028 | 0.02 |
| doctor | 0.0327 | 0.21 | shit | -0.0275 | 0.02 |
| boyfriend | 0.0322 | 0.02 | govern | -0.0239 | 0.12 |
| foundat | 0.0273 | 0.35 | att | -0.0239 | 0.35 |
| oh | 0.027 | 0.13 | yea | -0.0233 | 0.17 |
| color | 0.027 | 0.14 | guy | -0.0227 | 0.01 |
| woman | 0.0238 | 0.05 | war | -0.021 | 0.05 |
| cute | 0.0233 | 0.11 | gun | -0.021 | 0.07 |
| dog | 0.0227 | 0.1 | deck | -0.0199 | 0.12 |
| hair | 0.0225 | 0.05 | ass | -0.0197 | 0.12 |
| thank | 0.0222 | 0.04 | player | -0.0185 | 0.04 |
| skin | 0.0212 | 0.12 | point | -0.0181 | 0.01 |
| help | 0.0195 | 0.03 | countri | -0.0175 | 0.05 |

Senza entrambi:
| verso femminile | peso F | quota 1 utente F | verso maschile | peso M | quota 1 utente M |
|---|---|---|---|---|---|
| okay | 0.0633 | 0.21 | jon | -0.1412 | 0.93 |
| smile | 0.059 | 0.5 | fuck | -0.0595 | 0.07 |
| love | 0.0489 | 0.09 | man | -0.0338 | 0.06 |
| oh | 0.0479 | 0.19 | car | -0.0251 | 0.14 |
| femal | 0.0252 | 0.1 | transmiss | -0.0247 | 0.89 |
| dog | 0.0251 | 0.13 | govern | -0.0244 | 0.13 |
| thank | 0.025 | 0.05 | shit | -0.024 | 0.03 |
| doctor | 0.0239 | 0.29 | ass | -0.0239 | 0.17 |
| cute | 0.0222 | 0.16 | im | -0.0226 | 0.19 |
| lot | 0.0203 | 0.02 | yea | -0.0226 | 0.21 |
| woman | 0.0189 | 0.08 | gun | -0.0214 | 0.09 |
| feel | 0.0187 | 0.03 | team | -0.0179 | 0.04 |
| ask | 0.0181 | 0.08 | countri | -0.0179 | 0.06 |
| women | 0.0175 | 0.03 | form | -0.0175 | 0.16 |
| help | 0.0171 | 0.04 | world | -0.0166 | 0.08 |
| gender | 0.0161 | 0.06 | deck | -0.0161 | 0.15 |
| sorri | 0.0159 | 0.07 | vs | -0.0153 | 0.34 |
| tran | 0.0152 | 0.19 | batteri | -0.0152 | 0.3 |
| food | 0.0149 | 0.03 | guy | -0.0148 | 0.02 |
| nod | 0.0145 | 0.72 | war | -0.0145 | 0.07 |

Grafico: `figures/shortcut_scenari.png`.
