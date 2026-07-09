#!/usr/bin/env bash
#
# Reproducible PDF build for PAPER_DRAFT.md.
#
# Requirements:
#   - pandoc            (brew install pandoc)
#   - a LaTeX engine    (xelatex, e.g. via MacTeX / TeX Live)
#   - the "STIX Two Text" font (ships with recent macOS; otherwise install STIX Two)
#
# The source markdown is left untouched. A temporary copy is post-processed so
# that the handful of unicode math operators STIX Two Text does not carry are
# rendered as standard LaTeX math commands.
#
set -euo pipefail
cd "$(dirname "$0")"

SRC="PAPER_DRAFT.md"
OUT="RK4Optimizer_paper.pdf"
TMP=".paper_build.tmp.md"
trap 'rm -f "$TMP"' EXIT

# Promote the first H1 to the PDF title block; the source stays untouched.
TITLE="$(sed -n '1s/^# //p' "$SRC")"
tail -n +2 "$SRC" > "$TMP"

pandoc "$TMP" -o "$OUT" \
  --pdf-engine=xelatex \
  -H paper_glyphs.tex \
  --metadata title="$TITLE" \
  --toc --toc-depth=2 \
  -V geometry:margin=1in -V fontsize=11pt \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue -V toccolor=black \
  -V mainfont="STIX Two Text" \
  -V monofont="Menlo" -V monofontoptions="Scale=0.85"

echo "Wrote $OUT"
