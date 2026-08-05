# Build MaritimeRAG chunks + full_corpus_715_v1 index (run from anywhere)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Rag = Join-Path $Root "rag"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "data\processed\logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$Log = Join-Path $LogDir "build_rag_index.log"

function Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Add-Content -Path $Log -Value $line -Encoding UTF8
  Write-Host $line
}

Set-Location $Rag
$env:PYTHONUTF8 = "1"
$env:Path = "$(Join-Path $Root '.venv\Scripts');" + $env:Path

Log "=== START preprocess 715 docs ==="
& $VenvPython scripts/run_rag_batch.py `
  --manifest data/manifests/pdf_manifest.csv `
  --doc-list data/manifests/full_corpus_715_remaining.csv `
  --steps pdf,layout,merge,crop,chunks `
  --resume-completed `
  --skip-on-error *>> $Log 2>&1
Log "preprocess exit=$LASTEXITCODE"

Log "=== refresh remaining manifest ==="
& $VenvPython scripts/prepare_full_corpus_715.py `
  --manifest data/manifests/pdf_manifest.csv `
  --output data/manifests/full_corpus_715.csv `
  --remaining-output data/manifests/full_corpus_715_remaining.csv *>> $Log 2>&1

Log "=== START unified Chroma index full_corpus_715_v1 ==="
& $VenvPython scripts/10_build_unified_index.py `
  --doc-list data/manifests/full_corpus_715.csv `
  --manifest data/manifests/full_corpus_715.csv `
  --collection-id full_corpus_715_v1 `
  --embedding-preset e5-base `
  --include-types text,picture `
  --structured-tables exclude `
  --max-embedding-tokens 420 `
  --embedding-overlap-tokens 60 *>> $Log 2>&1
Log "unified index exit=$LASTEXITCODE"

Log "=== START BM25 index ==="
& $VenvPython scripts/35_build_bm25_index.py --unified full_corpus_715_v1 --rebuild *>> $Log 2>&1
Log "bm25 exit=$LASTEXITCODE"
Log "=== ALL DONE ==="
