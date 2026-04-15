#!/usr/bin/env bash
#
# Romantic Language Timeline: 50-Novel RLM Experiment
#
# Downloads 50 novels (5 per decade, 1800-1900) from Project Gutenberg,
# organized by decade in subdirectories, and runs an RLM analysis that
# tracks how romantic language evolves across the century.
#
# Designed to force genuine code-writing: chunking, word counting,
# frequency computation, sentiment aggregation, cross-document synthesis.
#
# See docs/superpowers/specs/2026-04-09-romantic-timeline-experiment-design.md
#
# Usage:
#   ./scripts/romantic_timeline_experiment.sh
#   ./scripts/romantic_timeline_experiment.sh --model gpt-5.4 --provider openai
#   ./scripts/romantic_timeline_experiment.sh --dry-run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RLM_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="/tmp/rlm_romantic_timeline"
RESULTS_DIR="$RLM_DIR/scripts/romantic_timeline_results"

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

# ── Novel corpus ─────────────────────────────────────────────────────────────
#
# 5 novels per decade, 1800-1900. Format: DECADE GUTENBERG_ID FILENAME
# Selection: English-language originals or major translations on Gutenberg.
# Mix of romance-heavy and romance-light per decade.

NOVELS=(
    # 1800s
    "1800s 161   sense_and_sensibility.txt"
    "1800s 121   northanger_abbey.txt"
    "1800s 1424  castle_rackrent.txt"
    "1800s 3268  the_mysteries_of_udolpho.txt"
    "1800s 9455  belinda.txt"

    # 1810s
    "1810s 1342  pride_and_prejudice.txt"
    "1810s 141   mansfield_park.txt"
    "1810s 158   emma.txt"
    "1810s 105   persuasion.txt"
    "1810s 7025  rob_roy.txt"

    # 1820s
    "1820s 82    ivanhoe.txt"
    "1820s 940   the_last_of_the_mohicans.txt"
    "1820s 9845  the_spy.txt"
    "1820s 7974  the_pilot.txt"
    "1820s 2275  the_pioneers.txt"

    # 1830s
    "1830s 1260  jane_eyre.txt"
    "1830s 768   wuthering_heights.txt"
    "1830s 2610  the_hunchback_of_notre_dame.txt"
    "1830s 25344 the_scarlet_letter.txt"
    "1830s 46    a_christmas_carol.txt"

    # 1840s
    "1840s 766   david_copperfield.txt"
    "1840s 1399  anna_karenina.txt"
    "1840s 599   vanity_fair.txt"
    "1840s 969   the_tenant_of_wildfell_hall.txt"
    "1840s 767   agnes_grey.txt"

    # 1850s
    "1850s 1400  great_expectations.txt"
    "1850s 98    a_tale_of_two_cities.txt"
    "1850s 2701  moby_dick.txt"
    "1850s 76    adventures_of_huckleberry_finn.txt"
    "1850s 219   heart_of_darkness.txt"

    # 1860s
    "1860s 135   les_miserables.txt"
    "1860s 2554  crime_and_punishment.txt"
    "1860s 2600  war_and_peace.txt"
    "1860s 514   little_women.txt"
    "1860s 35    the_time_machine.txt"

    # 1870s
    "1870s 541   the_age_of_innocence.txt"
    "1870s 209   the_turn_of_the_screw.txt"
    "1870s 122   the_return_of_the_native.txt"
    "1870s 143   the_mayor_of_casterbridge.txt"
    "1870s 107   far_from_the_madding_crowd.txt"

    # 1880s
    "1880s 174   the_picture_of_dorian_gray.txt"
    "1880s 244   a_study_in_scarlet.txt"
    "1880s 36    the_war_of_the_worlds.txt"
    "1880s 120   treasure_island.txt"
    "1880s 43    the_strange_case_of_dr_jekyll_and_mr_hyde.txt"

    # 1890s
    "1890s 110   tess_of_the_durbervilles.txt"
    "1890s 345   dracula.txt"
    "1890s 1952  the_yellow_wallpaper.txt"
    "1890s 1837  the_prince_and_the_pauper.txt"
    "1890s 160   the_awakening.txt"
)

# ── Download ─────────────────────────────────────────────────────────────────

echo "=== Downloading 50 novels to $DATA_DIR ==="

for entry in "${NOVELS[@]}"; do
    decade=$(echo "$entry" | awk '{print $1}')
    id=$(echo "$entry" | awk '{print $2}')
    filename=$(echo "$entry" | awk '{print $3}')
    dest_dir="$DATA_DIR/$decade"
    dest="$dest_dir/$filename"

    mkdir -p "$dest_dir"

    if [[ -f "$dest" ]]; then
        size=$(wc -c < "$dest" | tr -d ' ')
        echo "  [skip] $decade/$filename (${size} bytes)"
        continue
    fi

    url="https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt"
    echo "  [fetch] $decade/$filename"
    if ! curl -sS -L -o "$dest" "$url"; then
        echo "  [ERROR] Failed to download $decade/$filename" >&2
        rm -f "$dest"
        exit 1
    fi
    size=$(wc -c < "$dest" | tr -d ' ')
    echo "  [ok]    $decade/$filename (${size} bytes)"
done

total_bytes=$(du -sh "$DATA_DIR" | awk '{print $1}')
file_count=$(find "$DATA_DIR" -name "*.txt" | wc -l | tr -d ' ')
echo ""
echo "=== Download complete: $file_count files, $total_bytes ==="
echo ""

if $DRY_RUN; then
    echo "(dry-run mode — skipping RLM execution)"
    echo ""
    echo "Corpus layout:"
    for decade_dir in "$DATA_DIR"/*/; do
        decade=$(basename "$decade_dir")
        count=$(ls "$decade_dir" | wc -l | tr -d ' ')
        size=$(du -sh "$decade_dir" | awk '{print $1}')
        echo "  $decade: $count novels, $size"
    done
    exit 0
fi

# ── Run experiment ───────────────────────────────────────────────────────────

QUERY="Analyze the evolution of romantic language across 50 novels spanning 1800-1900. \
The novels are organized by decade in subdirectories.

For each novel:
1. Identify and extract passages containing romantic language (courtship, \
declarations of love, descriptions of attraction, romantic suffering, \
marriage proposals, etc.)
2. Compute metrics: frequency of romantic passages per 10,000 words, \
average sentiment intensity (1-5 scale), and the dominant romantic \
themes present.

Then synthesize across all novels:
- Build a decade-by-decade timeline showing how romantic language frequency, \
sentiment intensity, and dominant themes shift from 1800 to 1900.
- Identify the inflection points: which decades show the biggest changes \
and why?
- Name the top 5 most romantically intense passages across the entire corpus \
with exact quotes.

Return structured JSON with per-novel data, per-decade aggregates, the \
timeline, and the top passages."

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
    --max-iterations 25
    --save-transcript "$transcript_file"
)

if [[ -n "$MODEL" ]]; then
    CLI_ARGS+=(--model "$MODEL")
fi
if [[ -n "$PROVIDER" ]]; then
    CLI_ARGS+=(--provider "$PROVIDER")
fi
CLI_ARGS+=("${EXTRA_ARGS[@]}")

echo "=== Running Romantic Timeline experiment ==="
echo "  Model:      ${MODEL:-default}"
echo "  Provider:   ${PROVIDER:-default}"
echo "  Data:       $DATA_DIR (50 novels, 10 decades)"
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
