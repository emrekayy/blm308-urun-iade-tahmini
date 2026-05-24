#!/usr/bin/env bash
# BLM308 bonus LaTeX derleme betiği
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

compile_local() {
    local main="$1"
    pdflatex -interaction=nonstopmode "$main.tex" >/dev/null
    pdflatex -interaction=nonstopmode "$main.tex" >/dev/null
}

compile_docker() {
    local main="$1"
    docker run --rm \
        -v "$SCRIPT_DIR/..:/workdir" \
        -w /workdir/bonus \
        texlive/texlive:latest \
        sh -c "pdflatex -interaction=nonstopmode ${main}.tex && pdflatex -interaction=nonstopmode ${main}.tex"
}

if command -v pdflatex >/dev/null 2>&1; then
    echo "Derleme (yerel pdflatex): report.tex"
    compile_local report
    echo "Derleme (yerel pdflatex): presentation.tex"
    compile_local presentation
elif command -v docker >/dev/null 2>&1; then
    echo "Derleme (Docker texlive): report.tex"
    compile_docker report
    echo "Derleme (Docker texlive): presentation.tex"
    compile_docker presentation
else
    echo "Hata: pdflatex veya docker bulunamadı."
    echo "macOS: brew install --cask basictex && sudo tlmgr update --self --all"
    exit 1
fi

echo "Tamamlandı:"
ls -la report.pdf presentation.pdf 2>/dev/null || true
