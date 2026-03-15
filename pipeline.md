# Pipeline.md
## End-to-End Recommendation Pipeline (Visual + Deep Explanation)

## 1) Visual Workflow
```mermaid
flowchart TD
    A[MovieLens Data\nratings.csv + movies.csv] --> B[Data Loading\nPandas read_csv]
    B --> C[Train/Test Split\nrandom_state=42, test_size=0.2]
    C --> D[Baseline\nGlobal Mean Prediction]
    C --> E[User-Item Matrix\nPivot + Fill NaN with Global Mean]
    E --> F[NMF Training\nn_components=100, max_iter=200]
    F --> G[Reconstruction\nR_hat = W x H\nClip to [0.5, 5.0]]
    D --> H[Pointwise Eval\nRMSE, MAE]
    G --> H
    G --> I[Ranking Eval\nPrecision@10, Recall@10, F1@10, NDCG@10, MRR@10]
    G --> J[Top-N Recommendation\noptional: user_id=42]
    H --> K[Metrics Artifact\nreports/pipeline_metrics.json]
    I --> K
    J --> L[Recommendation Artifact (optional)\ncustom path via CLI]
    F --> M[Model Artifact\nmodels/nmf_pipeline.pkl]
```

## 2) Step-by-Step I/O
### Step A-B: Data Loading
- **Input**:
  - `data/ml-latest-small/ratings.csv`
  - `data/ml-latest-small/movies.csv`
- **Output**:
  - DataFrames for ratings and movie metadata.

### Step C: Train/Test Split
- **Input**: full ratings DataFrame.
- **Config**: `test_size=0.2`, `random_state=42`.
- **Output**:
  - `train_df` (`80,668` rows)
  - `test_df` (`20,168` rows)

### Step D: Baseline
- **Method**: predict all test ratings by train global mean.
- **Output**: baseline RMSE/MAE as reference.

### Step E-F-G: NMF Modeling
- Build user-item matrix from train set via pivot table.
- Fill missing values (`NaN`) using global mean.
- Train NMF with:
  - `n_components=100`
  - `max_iter=200`
  - `init='nndsvda'`
- Reconstruct rating matrix: \(\hat{R}=WH\), clip to `[0.5, 5.0]`.

### Step H-I: Evaluation
- **Pointwise metrics**: RMSE, MAE (test set).
- **Ranking metrics**: Precision@10, Recall@10, F1@10, NDCG@10, MRR@10.
- **Current metrics source-of-truth**: `reports/pipeline_metrics.json`.

### Step J-M: Artifact Export
- `models/nmf_pipeline.pkl`: trained model + factors + mappings.
- `reports/pipeline_metrics.json`: evaluation summary + params + improvement.
- Optional demo Top-N output: provide `--recommendations-out <path>` when needed.

## 3) Quality Control Points
1. **Data integrity**: verify counts and unique users/movies before split.
2. **Metric integrity**: always compare with baseline before claiming improvement.
3. **Reproducibility**: fixed random seed.
4. **Narrative consistency**: documentation must align with NMF metrics in artifacts.

## 4) Convergence / Loss Note (Honest Status)
- Current implementation does **not** export epoch-by-epoch training loss.
- Practical convergence evidence currently comes from final test metrics stability.
- Recommended convergence proxy experiment for presentation:
  - Run multiple `max_iter` values: `50, 100, 150, 200`
  - Compare RMSE/MAE trend to show fast/slow convergence behavior.

## 5) How to Visualize and Move to Assets
### Option A: Render Mermaid in VS Code
1. Open `pipeline.md`.
2. Use Markdown preview with Mermaid support.
3. Export screenshot to `assets/figures/pipeline_workflow.png`.

### Option B: GitHub-style visual capture
1. Push to branch.
2. Open `pipeline.md` in GitHub.
3. Capture clean screenshot and store in `assets/figures/`.

## 6) Embed In README
After exporting image, embed in README:
```md
![Pipeline Workflow](assets/figures/pipeline_workflow.png)
```

## 7) Key Submission Message
- Pipeline is reproducible, baseline-compared, and metric-driven.
- It is suitable for instructor review because all major claims are traceable to generated artifacts.
