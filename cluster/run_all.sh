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

# preprocessing.py e supervised_common.py non si eseguono: contengono solo funzioni
# Versioni delle librerie nel log (riproducibilita')
python3 -c "import sys, numpy, pandas, scipy, sklearn, matplotlib, nltk; print('Python', sys.version.split()[0], '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| scipy', scipy.__version__, '| scikit-learn', sklearn.__version__, '| matplotlib', matplotlib.__version__, '| nltk', nltk.__version__)"

for script in ML4N_phase1.py fase2_clean.py fase2a_bow.py fase2b_tfidf.py fase2c_pca.py \
              fase3a_dataset.py fase3b_modelli_default.py fase3c_overfitting.py fase3d_tuning.py fase3e_fasce.py \
              fase3f_verifiche.py fase3g_subreddit.py fase3h_shortcut.py; do
    echo "===== $(date '+%H:%M:%S') $script"
    python3 "$script"
done
echo "===== $(date '+%H:%M:%S') completato"
