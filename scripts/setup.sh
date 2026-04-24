#!/bin/bash
# Lore — full setup for macOS Apple Silicon
# Run from project root: ./scripts/setup.sh

set -e

echo "=== Lore Setup ==="
echo ""

# 1. Preflight checks
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install with: brew install python3"
    exit 1
fi

if ! command -v brew &>/dev/null; then
    echo "WARNING: Homebrew not found. System tools (yt-dlp, ffmpeg) won't auto-install."
    echo "  Install Homebrew: https://brew.sh"
    HAS_BREW=false
else
    HAS_BREW=true
fi

# 2. System tools (before Python — these are needed for video/audio ingest)
echo "System tools..."
for cmd in yt-dlp ffmpeg; do
    if command -v $cmd &>/dev/null; then
        echo "  $cmd: OK"
    elif $HAS_BREW; then
        echo "  $cmd: installing via brew..."
        brew install $cmd --quiet
    else
        echo "  $cmd: MISSING — install with: brew install $cmd"
    fi
done

# 3. Python venv
echo ""
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment exists."
fi

source .venv/bin/activate

# 4. Install package with all deps
echo ""
echo "Installing Lore + all dependencies..."
pip install -e ".[all]" --quiet

# 5. spaCy model
echo ""
echo "Downloading spaCy English model..."
python3 -m spacy download en_core_web_sm --quiet 2>/dev/null || python3 -m spacy download en_core_web_sm

# 6. .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file..."
    cat > .env << 'ENVEOF'
# OpenRouter API key for LLM enrichment (stages 2-4)
# Get one at https://openrouter.ai/keys
LORE_CUSTOM_API_KEY=
ENVEOF
    echo "  Created .env — add your OpenRouter API key for enrichment."
else
    echo ""
    echo ".env exists."
fi

# 7. Data directory
mkdir -p ~/.lore
echo ""
echo "Data directory: ~/.lore/"

# 8. Verify
echo ""
echo "=== Verification ==="
FAIL=false
python3 -c "
import sherpa_onnx; print(f'  sherpa-onnx {sherpa_onnx.__version__}: OK')
import lancedb; print('  LanceDB: OK')
import flashrank; print('  FlashRank: OK')
import keybert; print('  KeyBERT: OK')
import spacy; nlp = spacy.load('en_core_web_sm'); print('  spaCy + en_core_web_sm: OK')
import rapidfuzz; print('  rapidfuzz: OK')
import mcp; print('  MCP SDK: OK')
import trafilatura; print('  trafilatura: OK')
import pymupdf; print('  PyMuPDF: OK')
from bs4 import BeautifulSoup; print('  BeautifulSoup: OK')
import ebooklib; print('  ebooklib: OK')
import httpx; print('  httpx: OK')
" || FAIL=true

for cmd in yt-dlp ffmpeg; do
    if command -v $cmd &>/dev/null; then
        echo "  $cmd: OK"
    else
        echo "  $cmd: MISSING"
        FAIL=true
    fi
done

echo ""
if $FAIL; then
    echo "=== Setup incomplete — see errors above ==="
    exit 1
fi

echo "=== Setup complete ==="
echo ""
echo "Start the MCP server (stdio):  python -m lore --mcp-stdio"
echo "Start the HTTP server:         python -m lore"
echo "Run batch ingest:              python scripts/batch_ingest.py"
