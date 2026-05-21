#!/usr/bin/env bash
# Render docs/learn-vggt.html to docs/learn-vggt.pdf via headless Chrome.
# macOS-only path; adjust the Chrome path for other OSes.
set -euo pipefail
cd "$(dirname "$0")/.."

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [[ ! -x "$CHROME" ]]; then
  echo "Chrome not found at $CHROME" >&2
  exit 1
fi

"$CHROME" \
  --headless=new \
  --disable-gpu \
  --no-sandbox \
  --hide-scrollbars \
  --virtual-time-budget=15000 \
  --run-all-compositor-stages-before-draw \
  --no-pdf-header-footer \
  --print-to-pdf="$PWD/docs/learn-vggt.pdf" \
  "file://$PWD/docs/learn-vggt.html"

echo "Wrote $PWD/docs/learn-vggt.pdf ($(stat -f%z docs/learn-vggt.pdf) bytes, $(mdls -raw -name kMDItemNumberOfPages docs/learn-vggt.pdf 2>/dev/null || echo '?') pages)"
