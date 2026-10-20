#!/usr/bin/env bash
# Persistenta gratuita a bazei de date intre rulari GitHub Actions: un fisier (pzu.db.gz) atasat
# unui release (tag `data-latest`), suprascris la fiecare rulare. Necesita GH_TOKEN si `contents: write`.
#
#   scripts/ci_db.sh pull [--optional]     descarca DB (fara --optional: eroare daca nu exista)
#   scripts/ci_db.sh push [fisier ...]     urca DB doar daca s-a schimbat (+ fisiere extra, ex. dashboard)
set -euo pipefail
TAG=data-latest
ASSET=pzu.db.gz
DB=${DB:-pzu.db}

case "${1:-}" in
  pull)
    if gh release download "$TAG" -p "$ASSET" --output "$ASSET" --clobber 2>/dev/null; then
      gunzip -f "$ASSET"
      sha256sum "$DB" | cut -d' ' -f1 > "$DB.sha"
      echo "Baza de date descarcata din release ($(du -h "$DB" | cut -f1))."
    elif [ "${2:-}" = "--optional" ]; then
      echo "Nu exista inca o baza de date in release."
    else
      echo "::error::Nu exista baza de date in release '$TAG'. Ruleaza mai intai workflow-ul 'Bootstrap'."
      exit 1
    fi
    ;;
  push)
    shift
    [ -f "$DB" ] || { echo "Nu exista $DB; nu urc nimic."; exit 0; }
    gh release view "$TAG" >/dev/null 2>&1 || gh release create "$TAG" \
      --title "PZU data" --notes "Baza de date si dashboard, actualizate automat de GitHub Actions. Nu sterge." --latest=false
    new=$(sha256sum "$DB" | cut -d' ' -f1)
    if [ "$new" != "$(cat "$DB.sha" 2>/dev/null || true)" ]; then
      gzip -c "$DB" > "$ASSET"
      gh release upload "$TAG" "$ASSET" --clobber
      echo "Baza de date urcata ($(du -h "$ASSET" | cut -f1) comprimat)."
    else
      echo "Baza de date neschimbata; nu urc."
    fi
    if [ "$#" -gt 0 ]; then gh release upload "$TAG" "$@" --clobber; fi
    ;;
  *) echo "Utilizare: $0 pull [--optional] | push [fisiere...]"; exit 2 ;;
esac
