#!/usr/bin/env bash
# Fetch Marfoglia et al. (MOTU) public artifacts into ./_data/ (gitignored).
# Idempotent: skips files already present. See README.md.
set -euo pipefail
cd "$(dirname "$0")"
DATA="_data"
GL="https://gitlab.com/almahealthdb/ahdb-mapping-service/-/raw/master"
GLAPI="https://gitlab.com/api/v4/projects/almahealthdb%2Fahdb-mapping-service/repository"
mkdir -p "$DATA"/{sd,maps,shared,csv,output,validation,scripts}

echo "== GitLab: StructureDefinitions (logical models) + FML maps =="
for f in $(curl -sS "$GLAPI/tree?path=resources/fhir/StructureDefinition&per_page=100" | python3 -c "import json,sys;[print(e['path']) for e in json.load(sys.stdin) if e['type']=='blob']"); do
  curl -sS "$GL/$f" -o "$DATA/sd/$(basename "$f")"
done
for f in $(curl -sS "$GLAPI/tree?path=resources/mappings&per_page=100" | python3 -c "import json,sys;[print(e['path']) for e in json.load(sys.stdin) if e['path'].endswith('.map')]"); do
  curl -sS "$GL/$f" -o "$DATA/maps/$(basename "$f")"
done

echo "== GitLab: shared resources (Questionnaire/Organization/CodeSystem/ValueSet) =="
for sub in Questionnaire Organization CodeSystem ValueSet; do
  for f in $(curl -sS "$GLAPI/tree?path=resources/fhir/$sub&per_page=100" | python3 -c "import json,sys;[print(e['path']) for e in json.load(sys.stdin) if e['type']=='blob']" 2>/dev/null); do
    curl -sS "$GL/$f" -o "$DATA/shared/$(basename "$f")"
  done
done

echo "== GitLab: validation queries + cleaning scripts =="
curl -sS "$GL/validation/validation-queries-test.ipynb" -o "$DATA/validation/validation-queries-test.ipynb" || true
curl -sS "$GL/validation/bulk-data-export.py" -o "$DATA/validation/bulk-data-export.py" || true
for f in data_cleaning.ipynb data_cleaning.types.json data_cleaning.whitelist.json; do
  curl -sS "$GL/scripts/$f" -o "$DATA/scripts/$f" || true
done

echo "== Zenodo 10683153: input CSVs =="
for csv in Patient HospitalStay Fall Drug Knee; do
  [ -f "$DATA/csv/$csv.csv" ] || curl -sSL "https://zenodo.org/records/10683153/files/$csv.csv?download=1" -o "$DATA/csv/$csv.csv"
done

echo "== Zenodo 10684280: their FHIR output (ground truth) =="
[ -f "$DATA/output/dataset.zip" ] || curl -sSL "https://zenodo.org/records/10684280/files/dataset.zip?download=1" -o "$DATA/output/dataset.zip"
( cd "$DATA/output" && [ -d dataset ] || unzip -q -o dataset.zip || true )

echo "== done. tree: =="
find "$DATA" -maxdepth 2 -type f | sed 's/^/  /' | head -60
