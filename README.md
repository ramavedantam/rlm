# RLM: Recursive Language Model

An LLM with a Python REPL that can call itself recursively on sub-problems.

Based on: Zhang, Kraska, Khattab — ["Recursive Language Models"](https://alexzhang13.github.io/blog/2025/rlm/) (ICML 2025).

## Quick Start

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), an OpenAI API key (`OPENAI_API_KEY`).

```bash
git clone <repo-url> && cd rlm
./scripts/quickstart.sh
```

This downloads 5 short stories from Project Gutenberg (~230KB) and runs an RLM query that reads each story, identifies themes and emotional arcs, and ranks them by intensity — all with structured JSON output.

What you'll see (abbreviated):

```
--- Iteration 1 ---
Root LM: Let me explore the data directory first.
  files = list_files("/tmp/rlm_quickstart")
REPL stdout: ['adventures_of_sherlock_holmes.txt', 'alices_adventures_in_wonderland.txt', ...]

--- Iteration 2 ---
Root LM: I'll read each story and extract key information.
  stories = {}
  for f in list_files("/tmp/rlm_quickstart"):
      text = read_file(f"/tmp/rlm_quickstart/{f}")
      stories[f] = text[:5000]  # preview
REPL stdout: Read 5 stories...

--- Iteration 3 ---
Root LM: Now I'll analyze each story using llm_query for thematic classification.
  results = {}
  for name, text in stories.items():
      analysis = llm_query(f"Analyze this story: {text[:3000]}...")
      results[name] = analysis
...

--- Iteration 6 ---
Root LM: FINAL(result_json)

--- Stats ---
Iterations: 6
Input tokens: 45,231
Output tokens: 3,892
```

### Python API

```python
import asyncio
from rlm import OpenAIClient, RLM_REPL

async def main():
    client = OpenAIClient()  # uses OPENAI_API_KEY env var
    rlm = RLM_REPL(client=client, verbose=True)

    # Point it at a directory of files
    result = await rlm.run(
        query="Find the common themes across these stories and rank by intensity",
        context='DATA DIRECTORY: /tmp/stories\nUse list_files("/tmp/stories") to explore.',
        allowed_dirs=["/tmp/stories"],
    )

    print(result.answer)
    print(f"{result.iterations} iterations, {result.total_input_tokens + result.total_output_tokens:,} tokens")

    # Or pass data directly as a pre-loaded variable
    result = await rlm.run(
        query="Summarize the key findings",
        initial_vars={"data": {"users": 1500, "churn_rate": 0.12, "segments": [...]}},
    )

asyncio.run(main())
```

`RLM_REPL` is the core class. It takes an `LLMClient` (OpenAI or Anthropic), runs the REPL loop, and returns an `RLMResult` with `.answer`, `.iterations`, `.total_input_tokens`, `.total_output_tokens`, and `.code_log`.

## Three Ways to Pass Context

**Directory (folder of files):**
```bash
uv run rlm/cli.py -q "Analyze these logs" --context-dir ./logs/ --verbose
```
The model gets `list_files()` and `read_file()` access to the directory. The initial prompt includes a file listing. Best for multi-file analysis.

**Single file:**
```bash
uv run rlm/cli.py -q "Summarize this paper" --context-file paper.txt --verbose
```
Files under 50KB are inlined in the prompt. Larger files are accessible via `read_file()`. Best for single-document tasks.

**Inline string:**
```bash
uv run rlm/cli.py -q "Parse this data" --context-inline '{"users": [...]}' --verbose
```
Raw string injected directly into the prompt. Best for small payloads or piped input.

## What It Looks Like at Scale — The 50-Novel Experiment

**Stats:** 43MB corpus, 50 novels across 10 decades (1800s–1890s), 992K tokens, 11 iterations, ~330 lines of emergent Python.

We pointed the RLM at 50 novels from Project Gutenberg and asked it to track the evolution of romantic language across the 19th century. The model invented a three-stage pipeline without being told how:

### Stage 1: Code-based extraction

The model created a 60-keyword romance vocabulary with compiled regex, split all 50 novels into paragraphs, and filtered by keyword density:

```python
romance_keywords = [
    "love", "loved", "loving", "beloved", "adore", "adored", "adoration",
    "passion", "heart", "affection", "tender", "tenderness", "dear", "darling",
    "marry", "married", "marriage", "proposal", "propose", "engaged", "engagement",
    "kiss", "embrace", "embraced", "court", "courtship", "suitor", "woo", "wedded",
    "jealous", "jealousy", "devotion", "attachment", "admire", "admiration",
    "beauty", "beautiful", "charm", "charming", "attraction", "desire", "longing",
    "yearn", "yearning", "fond", "fondness", "lover", "lovers", "mistress", "bride",
    "bridegroom", "wedding", "union", "faithful", "constancy", "flame"
]

kw_re = re.compile(
    r'\b(?:' + '|'.join(
        re.escape(k) for k in sorted(set(romance_keywords), key=len, reverse=True)
    ) + r')\b', re.I
)

candidate_passages = {}
for key, text in clean_texts.items():
    paras = split_passages(text)
    hits = [p for p in paras if kw_re.search(p)]
    candidate_passages[key] = hits
```

### Stage 2: LLM-based classification

The model scored candidates with a heuristic (keyword density + dialogue bonus + length bonus), kept the top 80 per novel, and batched 10 novels at a time into `llm_query` calls with a controlled theme vocabulary:

```python
def passage_score_heuristic(p):
    text = p.lower()
    score = sum(text.count(k) for k in [
        "love", "marry", "marriage", "kiss", "beloved", "adore",
        "passion", "heart", "affection", "dear", "darling",
        "proposal", "engaged", "wedding", "jealous"
    ])
    if '"' in p or '\u201c' in p:
        score += 2  # dialogue bonus
    score += min(len(p.split()) // 40, 4)  # length bonus
    return score

# Top-80 pruning per novel, then batched LLM classification
for i in range(0, len(all_keys), batch_size):
    bk = all_keys[i:i+batch_size]
    batch_outputs.append(run_batch(bk))  # 10 novels per llm_query call
```

The controlled theme vocabulary is a key design choice: the model defined 10 theme categories and constrained the LLM to choose from them, ensuring the downstream aggregation code could count themes with `theme_counts[t] += 1`. The model recognized it was designing a data pipeline interface — not just asking a question.

### Stage 3: Code-based aggregation

All aggregation was code, not LLM — per-decade averages, decade-over-decade deltas, passage ranking:

```python
per_decade[d] = {
    "avg_romantic_frequency_per_10k_words": round(
        sum(r["romantic_frequency_per_10k_words"] for r in items) / len(items), 3
    ),
    "avg_sentiment_intensity_1_to_5": round(
        sum(r["avg_sentiment_intensity_1_to_5"] for r in items) / len(items), 3
    ),
    "dominant_themes": dominant,
}

# Decade-over-decade deltas
for i in range(1, len(decades)):
    prev = per_decade[decades[i-1]]
    cur = per_decade[decades[i]]
    changes.append({
        "frequency_change": round(
            cur["avg_romantic_frequency_per_10k_words"]
            - prev["avg_romantic_frequency_per_10k_words"], 3
        ),
        "sentiment_change": round(
            cur["avg_sentiment_intensity_1_to_5"]
            - prev["avg_sentiment_intensity_1_to_5"], 3
        ),
    })
```

### Results: Decade Timeline

| Decade | Avg Freq/10K | Avg Intensity | Dominant Themes |
|--------|:------------:|:-------------:|-----------------|
| 1800s | 1.98 | 3.70 | courtship, declaration of love, romantic suffering |
| 1810s | 1.82 | 3.96 | courtship, declaration of love, marriage/proposal |
| 1820s | 0.89 | 3.80 | courtship, declaration of love, loss/separation |
| 1830s | 1.89 | 4.30 | declaration of love, romantic suffering, loss/separation |
| 1840s | 1.46 | 4.02 | romantic suffering, courtship, loss/separation |
| 1850s | 1.25 | 3.56 | declaration of love, idealization, marriage/proposal |
| 1860s | 0.75 | 3.68 | attraction/beauty, declaration of love, loss/separation |
| 1870s | 1.62 | 3.72 | declaration of love, marriage/proposal, idealization |
| 1880s | 1.70 | 2.55 | marriage/proposal, attraction/beauty, courtship |
| 1890s | 2.14 | 3.52 | declaration of love, loss/separation, romantic suffering |

**Inflection points:** The 1820s dip (Austen → Cooper; romance gives way to adventure fiction), the 1830s Gothic/Romantic peak (Jane Eyre 4.4, Wuthering Heights 4.6, Hunchback of Notre-Dame 4.5), and the 1880s tonal collapse (Jekyll & Hyde, Treasure Island, War of the Worlds — genre fiction where romance is incidental).

### Top Passages

> "This reflection overcame Valancourt with tenderness, but, relapsing into despondency, he again felt only for himself, and lamented again this cruel separation, in a voice and words so impassioned, that Emily could no longer struggle to repress her own grief, or to sooth his."
> — *The Mysteries of Udolpho* (1800s)

> "Yes, Edward Weston, I could indeed be happy in a house full of enemies, if I had but one friend, who truly, deeply, and faithfully loved me; and if that friend were you..."
> — *Agnes Grey* (1840s)

Full analysis: [`scripts/romantic_timeline_results/run_20260409_133046_analysis.md`](scripts/romantic_timeline_results/run_20260409_133046_analysis.md)

## How It Works

**The loop.** The root LM writes Python in ` ```repl``` ` blocks. Code executes in a sandbox with persistent state. The model sees stdout/stderr and iterates until it calls `FINAL(answer)`.

**The tools.** Six functions injected into the REPL namespace:

| Function | Description |
|----------|-------------|
| `llm_query(prompt, system="")` | Single-shot LLM call, returns string |
| `rlm_query(query, context=None, context_dir=None)` | Spawn a child RLM with its own REPL loop |
| `read_file(path)` | Read from allowed directories |
| `list_files(directory)` | List files in allowed directories |
| `FINAL(answer)` | Submit final answer (dicts/lists auto-serialized to JSON) |
| `print(...)` | Output visible to model in next iteration |

**The key difference from agents.** In Claude Code / ReAct agents, the LLM orchestrates tools: think → call tool → see result → think → call tool. Every step passes through the LLM. In RLM, code orchestrates the LLM: the model writes a program that calls the LLM as a subroutine where judgment is needed. The for loop is in code, not in the agent loop.

```python
# RLM: code orchestrates LLM
for novel in novels:           # code loop, CPU speed
    passages = extract(novel)  # code — regex, no LLM needed
    scored = llm_query(passages)  # LLM as subroutine
    results.append(aggregate(scored))  # code
```

From the [blog post](https://alexzhang13.github.io/blog/2025/rlm/): "LMs should decide how to break down a problem."

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `-q, --query` | required | The query to answer |
| `--context-dir` | — | Directory of files to analyze |
| `--context-file` | — | Single file to use as context |
| `--context-inline` | — | Inline context string |
| `--model` | `gpt-5.4` | Root LM model |
| `--sub-model` | same as `--model` | Sub-LM model for `llm_query`/`rlm_query` |
| `--provider` | `openai` | LLM provider (`openai`, `anthropic`) |
| `--sub-provider` | same as `--provider` | Sub-LM provider |
| `--max-iterations` | `15` | Max REPL iterations |
| `--system-prompt-file` | — | Override system prompt |
| `-v, --verbose` | off | Print iteration details to stderr |
| `--json` | off | Output as JSON (answer + token stats) |
| `--save-transcript` | — | Save full transcript + code log to JSON |

## Experiments

**10-Novel Character Analysis** (`scripts/gutenberg_experiment.sh`): 10 classic novels (16MB), extract character networks and central conflicts. 2.7M tokens, 10/10 success rate. War and Peace identified as most complex character web (80 characters, 62 relationships). Emergent behaviors: Gutenberg header stripping, alias resolution (Russian patronymics, Spanish honorifics), relationship typing taxonomy.

**50-Novel Romantic Timeline** (`scripts/romantic_timeline_experiment.sh`): 50 novels across 1800–1900 (43MB), track romantic language evolution. 992K tokens, three-stage emergent pipeline. See full analysis in [`scripts/romantic_timeline_results/run_20260409_133046_analysis.md`](scripts/romantic_timeline_results/run_20260409_133046_analysis.md).

## References

- Zhang, Kraska, Khattab. "Recursive Language Models." ICML 2025.
- Blog: https://alexzhang13.github.io/blog/2025/rlm/
