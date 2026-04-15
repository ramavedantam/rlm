#!/usr/bin/env bash
#
# Gutenberg 10-Novel RLM Experiment
#
# Downloads 10 classic novels from Project Gutenberg and runs an RLM analysis
# that spawns a child rlm_query per novel to extract characters, relationships,
# and central conflicts, then synthesizes which novel has the most complex
# character web.
#
# Each run produces three artifacts in gutenberg_results/:
#   run_<ts>.json              — CLI JSON output (answer + token stats)
#   run_<ts>_transcript.json   — full transcript (config, messages, result)
#   run_<ts>.log               — verbose stderr log
#
# To analyze a run's transcript, point an LLM or agent at the transcript file.
#
# Usage:
#   ./scripts/gutenberg_experiment.sh              # defaults: gpt-5.4, openai
#   ./scripts/gutenberg_experiment.sh --model gpt-5.4 --provider openai
#   ./scripts/gutenberg_experiment.sh --dry-run     # download only, don't run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RLM_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="/tmp/rlm_gutenberg_test"
RESULTS_DIR="$RLM_DIR/scripts/gutenberg_results"

# Defaults (passthrough to cli.py)
MODEL=""
PROVIDER=""
EXTRA_ARGS=()
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)     MODEL="$2"; shift 2 ;;
        --provider)  PROVIDER="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Download novels ──────────────────────────────────────────────────────────

# Project Gutenberg plain-text URLs (UTF-8).
# Format: GUTENBERG_ID  FILENAME
NOVELS=(
    "1342   pride_and_prejudice.txt"
    "2701   moby_dick.txt"
    "2600   war_and_peace.txt"
    "135    les_miserables.txt"
    "2554   crime_and_punishment.txt"
    "1399   anna_karenina.txt"
    "996    don_quixote.txt"
    "1400   great_expectations.txt"
    "1727   the_odyssey.txt"
    "98     a_tale_of_two_cities.txt"
)

mkdir -p "$DATA_DIR"

echo "=== Downloading 10 novels to $DATA_DIR ==="
for entry in "${NOVELS[@]}"; do
    id=$(echo "$entry" | awk '{print $1}')
    filename=$(echo "$entry" | awk '{print $2}')
    dest="$DATA_DIR/$filename"

    if [[ -f "$dest" ]]; then
        size=$(wc -c < "$dest" | tr -d ' ')
        echo "  [skip] $filename (${size} bytes, already exists)"
        continue
    fi

    url="https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt"
    echo "  [fetch] $filename from $url"
    if ! curl -sS -L -o "$dest" "$url"; then
        echo "  [ERROR] Failed to download $filename" >&2
        rm -f "$dest"
        exit 1
    fi
    size=$(wc -c < "$dest" | tr -d ' ')
    echo "  [ok]    $filename (${size} bytes)"
done

total_bytes=$(du -sh "$DATA_DIR" | awk '{print $1}')
file_count=$(ls "$DATA_DIR" | wc -l | tr -d ' ')
echo ""
echo "=== Download complete: $file_count files, $total_bytes ==="
echo ""

if $DRY_RUN; then
    echo "(dry-run mode — skipping RLM execution)"
    exit 0
fi

# ── Run experiment ───────────────────────────────────────────────────────────

QUERY="For EACH of the 10 novels in this directory, use rlm_query to do a deep \
analysis. For each novel: identify every named character, map their relationships \
to each other, and identify the central conflict. After analyzing all 10, \
determine which novel has the most complex character web and explain why. \
Return a structured JSON result with all novels' analyses."

mkdir -p "$RESULTS_DIR"
timestamp=$(date +%Y%m%d_%H%M%S)
result_file="$RESULTS_DIR/run_${timestamp}.json"
transcript_file="$RESULTS_DIR/run_${timestamp}_transcript.json"
log_file="$RESULTS_DIR/run_${timestamp}.log"

CLI_ARGS=(
    -q "$QUERY"
    --context-dir "$DATA_DIR"
    --verbose
    --json
    --save-transcript "$transcript_file"
)

if [[ -n "$MODEL" ]]; then
    CLI_ARGS+=(--model "$MODEL")
fi
if [[ -n "$PROVIDER" ]]; then
    CLI_ARGS+=(--provider "$PROVIDER")
fi
CLI_ARGS+=("${EXTRA_ARGS[@]}")

echo "=== Running RLM experiment ==="
echo "  Model:      ${MODEL:-default}"
echo "  Provider:   ${PROVIDER:-default}"
echo "  Data:       $DATA_DIR"
echo "  Result:     $result_file"
echo "  Transcript: $transcript_file"
echo "  Log:        $log_file"
echo ""

cd "$RLM_DIR"
uv run rlm/cli.py "${CLI_ARGS[@]}" \
    > "$result_file" \
    2> >(tee "$log_file" >&2)

echo ""
echo "=== Experiment complete ==="
echo "  Result:     $result_file"
echo "  Transcript: $transcript_file"
echo "  Log:        $log_file"
