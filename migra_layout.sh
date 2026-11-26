#!/bin/bash
# Sposta nel nuovo layout i file NON versionati (dati, CSV preprocessati, matrici, split, log) rimasti in cluster/
# dopo il "git pull" che rinomina cluster/ in codice/. Da eseguire UNA volta, dalla cartella principale del repository.
# Non riesegue nulla e non cancella dati: se un file esiste gia' nella nuova posizione, lo lascia dov'e' e lo segnala.
#
# Uso:
#   cd ~/ML4N-project
#   git pull
#   bash migra_layout.sh
set -euo pipefail

[ -d codice ] || { echo "Manca la cartella codice/: eseguire prima git pull"; exit 1; }
[ -d cluster ] || { echo "La cartella cluster/ non esiste: niente da spostare"; exit 0; }

lasciati=0
sposta() {   # sposta() <cartella vecchia> <cartella nuova> [profondita' massima]
    local da=$1 a=$2 prof=${3:-100}
    [ -d "$da" ] || return 0
    while IFS= read -r -d '' f; do
        local rel=${f#"$da"/}
        local dest="$a/$rel"
        mkdir -p "$(dirname "$dest")"
        if [ -e "$dest" ]; then
            echo "gia' presente, lasciato in $f"; lasciati=$((lasciati + 1))
        else
            mv "$f" "$dest"
        fi
    done < <(find "$da" -maxdepth "$prof" \( -type f -o -type l \) -not -path "*/__pycache__/*" -print0)
}

echo "== Sezione 1: vettorizzazione, preprocessing, esplorazione"
sposta cluster/risultati_fase2/vettorizzazione codice/risultati_sezione1/vettorizzazione
sposta cluster/risultati_fase2 codice/risultati_sezione1/preprocessing
sposta cluster/risultati_fase1 codice/risultati_sezione1/esplorazione
echo "== Sezione 2"
sposta cluster/risultati_fase3 codice/risultati_sezione2
echo "== Dati e log nella cartella principale di cluster/"
sposta cluster codice 1

# La pipeline salvata nel 2.2c contiene la classe TfidfSlide, che prima stava in supervised_common.py e ora in
# comune.py: la si rilegge con il vecchio nome e la si salva di nuovo, cosi' punta al modulo nuovo.
if [ -f "$HOME/ml4n-env/bin/activate" ]; then source "$HOME/ml4n-env/bin/activate"; fi
python3 - <<'EOF'
import sys
sys.path.insert(0, "codice")
import joblib
import comune
sys.modules["supervised_common"] = comune
percorso = "codice/risultati_sezione2/pipeline_migliore.joblib"
joblib.dump(joblib.load(percorso), percorso)
print("pipeline_migliore.joblib aggiornata: ora usa comune.TfidfSlide")
EOF

find cluster -type d -name __pycache__ -prune -exec rm -rf {} +
find cluster -type d -empty -delete 2>/dev/null || true
if [ -d cluster ]; then
    echo "Restano in cluster/ questi file (da controllare a mano):"
    find cluster -type f
else
    echo "cluster/ svuotata e rimossa."
fi
echo "File lasciati perche' gia' presenti: $lasciati"
echo "Ora: git status (deve risultare modificata solo pipeline_migliore.joblib), poi commit e push."
