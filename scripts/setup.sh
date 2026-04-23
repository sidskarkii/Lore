#!/bin/bash
# Lore — full setup for macOS Apple Silicon
# Run from project root: ./scripts/setup.sh

set -e

echo "=== Lore Setup ==="
echo ""

# 1. Python venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment exists."
fi

source .venv/bin/activate

# 2. Install package with all deps
echo ""
echo "Installing Lore + all dependencies..."
pip install -e ".[all]" --quiet

# 3. spaCy model
echo ""
echo "Downloading spaCy English model..."
python3 -m spacy download en_core_web_sm --quiet 2>/dev/null || python3 -m spacy download en_core_web_sm

# 4. System tools
echo ""
echo "Checking system tools..."
for cmd in yt-dlp ffmpeg; do
    if command -v $cmd &>/dev/null; then
        echo "  $cmd: OK"
    else
        echo "  $cmd: MISSING — install with: brew install $cmd"
    fi
done

# 5. .env file
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

# 6. Data directory
mkdir -p ~/.lore
echo ""
echo "Data directory: ~/.lore/"

# 7. Verify
echo ""
echo "=== Verification ==="
python3 -c "
from faster_whisper import WhisperModel; print('  faster-whisper: OK')
import lancedb; print('  LanceDB: OK')
import flashrank; print('  FlashRank: OK')
import keybert; print('  KeyBERT: OK')
import spacy; print('  spaCy: OK')
import rapidfuzz; print('  rapidfuzz: OK')
import mcp; print('  MCP SDK: OK')
import trafilatura; print('  trafilatura: OK')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start the MCP server (stdio):  python -m lore --mcp-stdio"
echo "Start the HTTP server:         python -m lore"
echo "Run batch ingest:              python scripts/batch_ingest.py"
