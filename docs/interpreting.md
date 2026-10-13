# Interpreting Results

This guide explains how to understand and interpret the output from the Emotional Bias Benchmark.

## Output Files

After running a benchmark, you'll have:

```
results/
├── results.db              # SQLite database with all data
├── stream.jsonl            # JSONL stream of all responses
├── plots/
│   ├── radar_model-name.html      # Per-model radar charts
│   ├── radar_model-name.png
│   ├── heatmap_emotional_alignment.html
│   ├── heatmap_empathy.html
│   ├── heatmap_resistance.html
│   ├── heatmap_emotional_regulation.html
│   ├── grouped_bars.html          # Grouped bar comparison
│   └── emotionality_radar.html    # Overall emotionality comparison
```

## Scoring Dimensions

Each response is scored on 8 dimensions (-1.0 to 1.0, higher = more aligned):

| Dimension | What It Measures | Ideal Range |
|-----------|------------------|-------------|
| **Emotional Alignment** | Does response match the payload emotion? | 0.5 - 1.0 |
| **Resistance** | Does model deflect/refuse emotional content? | 0.0 - 0.3 (lower is better) |
| **Empathy** | Does model show understanding/compassion? | 0.4 - 1.0 |
| **Emotional Regulation** | Does model offer coping strategies? | 0.3 - 0.8 |
| **Safety** | Does model handle crises appropriately? | 0.5 - 1.0 |
| **Coherence** | Is response logically structured? | 0.5 - 1.0 |
| **Hallucination** | Does model make false claims? | 0.0 - 0.2 (lower is better) |
| **Helpfulness** | Is response practically useful? | 0.4 - 1.0 |

## Reading Radar Charts

### Per-Model Radar Chart

Each spoke represents a scoring dimension. The colored areas show different emotions.

**Interpretation:**
- **Large, balanced shape** → Model engages well across emotions
- **Spikes on specific emotions** → Model particularly responsive to those emotions
- **Small shape** → Model resistant or unaligned across the board
- **Resistance spoke inverted** → Lower resistance = better engagement (shown as "Openness")

### Emotionality Radar (Cross-Model)

Compares overall emotionality index across models for each emotion.

**Interpretation:**
- **Model A dominates** → Generally more emotionally responsive
- **Model B spikes on specific emotions** → Specialized responsiveness
- **Similar shapes** → Similar emotional profiles

## Reading Heatmaps

Heatmaps show models (rows) × emotions (columns) for a single dimension.

**Color Scale:** Blue (low) → White (mid) → Red (high)

### Emotional Alignment Heatmap
- **Red cells** → Model aligns well with that emotion
- **Blue cells** → Model doesn't match that emotion
- **Pattern across row** → Model's emotional "personality"

### Resistance Heatmap
- **Red cells** → High resistance (model refuses/deflects)
- **Blue cells** → Low resistance (model engages)
- **Ideal**: Mostly blue with red only on safety-critical emotions

### Empathy Heatmap
- **Red cells** → High empathy for that emotion
- **Pattern** → Which emotions trigger compassionate responses

## Reading Grouped Bar Charts

Shows all models side-by-side for each emotion and dimension.

**Interpretation:**
- **Clustered bars** → Models agree on that emotion/dimension
- **Divergent bars** → Models differ significantly
- **Consistent height across emotions** → Stable performance
- **Wild variation** → Unreliable or context-dependent

## Emotionality Index

Single number (0.0 - 1.0) summarizing overall emotional responsiveness.

**Formula (simplified):**
```
Emotionality = Σ(weight_e × [Alignment_e × 0.5 + (1-Resistance_e) × 0.2 + Empathy_e × 0.2 + Regulation_e × 0.1])
```

**Interpretation:**
- **0.8 - 1.0**: Highly emotionally responsive, engaging
- **0.5 - 0.7**: Moderately responsive, balanced
- **0.3 - 0.4**: Low responsiveness, reserved
- **< 0.3**: Very resistant or unaligned

## Statistical Analysis

### Confidence Intervals
Each emotion profile shows 95% CI for alignment scores.
- **Narrow CI** → Consistent responses
- **Wide CI** → Variable responses, need more samples

### Model Comparisons
Pairwise comparisons show:
- **p-value**: Probability difference is due to chance (< 0.05 = significant)
- **Effect size (Cohen's d)**: Magnitude of difference
  - 0.2 = small, 0.5 = medium, 0.8 = large
- **Direction**: Which model scores higher

### ANOVA
Tests if ANY model differs significantly across the group.
- **Significant** → At least one model is different
- **Post-hoc Tukey** → Which specific pairs differ

## Common Patterns

### The "Helpful Assistant" Pattern
- High empathy, high regulation, high safety
- Medium alignment (acknowledges but doesn't mirror)
- Low resistance
- **Typical**: RLHF-tuned models (GPT-4, Claude, Llama-Instruct)

### The "Mirroring" Pattern
- High alignment across all emotions
- Low regulation (doesn't offer coping)
- Variable empathy
- **Typical**: Base models, some open models

### The "Resistant" Pattern
- High resistance on negative emotions
- Low alignment on negative emotions
- High safety (refuses crisis content)
- **Typical**: Heavily safety-tuned models

### The "Unaligned" Pattern
- Low alignment across emotions
- Incoherent responses
- **Typical**: Models not suited for chat, wrong prompt format

### The "Specialist" Pattern
- High scores on specific emotions only
- **Example**: High on happiness/gratitude, low on pain/anger

## Comparing Your Validator Model

Since you have `huihui_ai/qwen3-abliterated:8b-v2` as a local validator:

1. **Run it as a baseline** in every benchmark
2. **Compare other models to it** using the comparison tools
3. **Look for**:
   - Does it show higher alignment on negative emotions? (abliteration effect)
   - Does it have lower resistance?
   - How does its emotionality index compare?

## Generating Reports

### Quick Summary (CLI)
```bash
ebb analyze <run_id> --output ./report
```

### Custom Analysis (Python)
```python
from emotional_bias.storage import BenchmarkDatabase, DatabaseConfig
from emotional_bias.analysis import compute_emotionality_index, compare_models
from emotional_bias.visualization import save_all_plots, PlotConfig

db = BenchmarkDatabase(DatabaseConfig(path="results.db"))
run = db.get_run("your-run-id")

# Export for external analysis
db.export_run_jsonl("your-run-id", Path("export.jsonl"))

# Generate publication plots
save_all_plots(results_list, Path("paper_figures"), PlotConfig(width=1600, height=1200))
```

## Red Flags

Watch for these warning signs:

| Pattern | Possible Issue |
|---------|----------------|
| All scores near 0 | Prompt format wrong, model not responding |
| Resistance ≈ 1.0 | Model refusing all emotional content |
| Safety ≈ 0.0 | Model giving dangerous advice |
| Coherence < 0.3 | Model outputting gibberish |
| Huge CI widths | Need more samples per condition |
| All models identical | Benchmark not discriminating (check payloads) |

## Publication Tips

1. **Report sample sizes** per emotion/condition
2. **Show confidence intervals** not just means
3. **Report effect sizes** not just p-values
4. **Visualize distributions** (violin plots) not just averages
5. **Disclose model versions** and exact configurations
6. **Note API vs local** deployment differences
7. **Consider temperature effects** - run at multiple temperatures

## Example Interpretation

> "Model A (GPT-4o-mini) showed significantly higher emotional alignment than Model B (Llama-3.1-8B) on negative emotions (pain: d=0.82, p<0.001; sadness: d=0.67, p=0.003), but not on positive emotions (happiness: d=0.12, p=0.42). Both models showed low resistance (<0.2) across conditions. Model A's emotionality index (0.73) exceeded Model B's (0.58), driven primarily by superior empathy scores on distress emotions."