#!/usr/bin/env bash
#
# RLM Quick Start: download 5 short stories, run an RLM analysis.
#
# Usage:
#   ./scripts/quickstart.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RLM_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="/tmp/rlm_quickstart"

# 5 short, well-known public domain stories from Project Gutenberg
STORIES=(
    "1952  the_yellow_wallpaper.txt"       # Charlotte Perkins Gilman (~50KB, psychological)
    "2148  the_tell_tale_heart.txt"         # Edgar Allan Poe (~15KB, horror)
    "7256  the_gift_of_the_magi.txt"        # O. Henry (~15KB, sentimental)
    "1661  adventures_of_sherlock_holmes.txt" # Arthur Conan Doyle (~120KB, detective)
    "11    alices_adventures_in_wonderland.txt" # Lewis Carroll (~30KB, fantasy)
)

echo "=== RLM Quick Start ==="
echo "Downloading 5 stories to $DATA_DIR..."
echo ""

mkdir -p "$DATA_DIR"

for entry in "${STORIES[@]}"; do
    id=$(echo "$entry" | awk '{print $1}')
    filename=$(echo "$entry" | awk '{print $2}')
    dest="$DATA_DIR/$filename"

    if [[ -f "$dest" ]]; then
        echo "  [cached] $filename"
        continue
    fi

    url="https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt"
    echo "  [fetch]  $filename"
    curl -sS -L --fail -o "$dest" "$url" 2>/dev/null || {
        echo "  [ERROR]  Failed to download $filename" >&2
        rm -f "$dest"
        exit 1
    }
done

total=$(du -sh "$DATA_DIR" | awk '{print $1}')
echo ""
echo "Corpus ready: 5 stories, $total"
echo ""

QUERY="Read each story and identify:
1. The central theme
2. The emotional arc (how does the emotional tone shift from beginning to end?)
3. A single representative quote that captures the story's essence

Then rank all 5 stories by emotional intensity (most intense first) with a brief justification.

Return structured JSON with per-story analysis and the final ranking."

echo "=== Running RLM ==="
echo ""

cd "$RLM_DIR"
uv run rlm/cli.py \
    -q "$QUERY" \
    --context-dir "$DATA_DIR" \
    --verbose
