#!/bin/bash
# Esegue in ordine tutti gli script di codice/. Si ferma al primo errore.
# Uso, dalla cartella codice/:
#   nohup ./run_all.sh > log_run_all.txt 2>&1 &
#   tail -f log_run_all.txt
set -e
cd "$(dirname "$0")"

# Ambiente con nltk (vedi setup del cluster)
if [ -f "$HOME/ml4n-env/bin/activate" ]; then
    source "$HOME/ml4n-env/bin/activate"
fi

# preprocessing.py e comune.py non si eseguono: contengono solo funzioni
# Versioni delle librerie nel log (riproducibilita')
python3 -c "import sys, numpy, pandas, scipy, sklearn, matplotlib, nltk; print('Python', sys.version.split()[0], '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| scipy', scipy.__version__, '| scikit-learn', sklearn.__version__, '| matplotlib', matplotlib.__version__, '| nltk', nltk.__version__)"

for script in s1_1_esplorazione.py s1_2_preprocessing.py s1_3_bow.py s1_3_tfidf.py s1_4_pca.py \
              s2_1_dataset.py s2_2a_default.py s2_2b_overfitting.py s2_2c_tuning.py s2_2d_fasce.py \
              s2_2_verifiche.py s2_3a_subreddit.py s2_3b_shortcut.py; do
    echo "===== $(date '+%H:%M:%S') $script"
    python3 "$script"
done
echo "===== $(date '+%H:%M:%S') completato"
