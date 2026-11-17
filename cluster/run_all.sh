#!/bin/bash
# Esegue in ordine tutti gli script di cluster/. Si ferma al primo errore.
# Uso, dalla cartella cluster/:
#   nohup ./run_all.sh > log_run_all.txt 2>&1 &
#   tail -f log_run_all.txt
set -e
cd "$(dirname "$0")"

# Ambiente con nltk (vedi setup del cluster)
if [ -f "$HOME/ml4n-env/bin/activate" ]; then
    source "$HOME/ml4n-env/bin/activate"
fi

# preprocessing.py non si esegue: contiene solo funzioni, importate da fase2_clean.py
for script in ML4N_phase1.py fase2_clean.py fase2a_bow.py fase2b_tfidf.py fase2c_pca.py; do
    echo "===== $(date '+%H:%M:%S') $script"
    python3 "$script"
done
echo "===== $(date '+%H:%M:%S') completato"
