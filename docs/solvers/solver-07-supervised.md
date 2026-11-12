# Solver 7 — Supervised learning player

**Files:** `solvers/supervised/{dataset,models,player}.py`  ·  **Depends on:** solver 2 (exact labels) and/or solver 5 (`mc_label`), features

## Purpose

The **inductive** solver (1.pdf, 17.pdf): learn a position evaluator from labelled examples instead of hand-coding knowledge. Compare several classifiers and regressors from the course. The best model plays directly, and can also be plugged into solver 3 as its evaluator.

## Techniques

| Technique | Source |
|-----------|--------|
| Supervised learning: training set of (feature vector, label) | 17.pdf |
| Feature extraction: hand features `phi` vs raw edge vector | 17.pdf |
| Input encoding: one 0/1 input per edge (distributed encoding) | 17.pdf |
| k-Nearest Neighbour | 17.pdf |
| Perceptron / linear classifier; sigmoid unit trained by gradient descent | 17.pdf |
| Feed-forward neural network (MLP) trained with backpropagation | 17.pdf |
| SVM, Naive Bayes, decision tree (comparison baselines, via scikit-learn) | 17.pdf (named) |
| Train / validation / test split, early stopping, L2 weight penalty | 17.pdf |
| Bias–variance, underfitting / overfitting | 17.pdf |
| CNN over the edge grid (optional) `[outside]` | — |

## Data (`dataset.py`)

| Source | Boards | Label |
|--------|--------|-------|
| Solver 2 exact table | 2×2, 2×3 (3×3 if solved) | exact `V` (net boxes for the player to move) |
| Solver 5 `mc_label` | 3×3, 4×4 | MC estimate of P(player to move wins), e.g. 200 greedy rollouts |

- **Position sampling:** random self-play games between mixed players (random / greedy / search), sampling positions at all stages. Deduplicate by canonical form. Target 50k–200k positions per board.
- **Task variants:**
  - regression: `V / boxes_remaining` scaled to `[-1, 1]`;
  - classification: winning vs not winning for the player to move.
- **Inputs:**
  - (a) `phi(s)` — 12 features;
  - (b) the raw edge vector (E bits) plus `to_move` and `first_player`;
  - (c) both concatenated.
- **Split:** 70 / 15 / 15 by game, not by position, to avoid leakage.

## Models (`models.py`)

| Model | Library | Notes |
|-------|---------|-------|
| kNN | sklearn | k ∈ {1, 3, 5, 9, 15}; inputs scaled |
| Perceptron | numpy (from scratch) | perceptron training rule from 17.pdf, classification only |
| Logistic / sigmoid unit | numpy (from scratch) | gradient descent on squared error or cross-entropy |
| MLP | torch | 1–2 hidden layers; ReLU or sigmoid; Adam or SGD; L2; early stopping on validation |
| SVM, Naive Bayes, decision tree | sklearn | comparison baselines only |
| CNN (optional) | torch | edges laid out on a `(2m+1) × (2n+1)` grid |

Write the perceptron and sigmoid-unit training **from scratch**, matching the slides' update rules. The report should show you implemented the course algorithms rather than only calling a library.

## Playing (`player.py`)

One-ply lookahead with extra-turn handling:

```
score(a) = k + V̂(s')   if move a completes k > 0 boxes
score(a) = −V̂(s')      otherwise
```

Play the argmax. Alternatively, pass `V̂` as the `evaluator` to `SearchPlayer` (solver 3), giving variant `search-learned`.

## Interface

```python
class LearnedPlayer(Player):
    name = "learned-<model>"
    def __init__(self, model_path, lookahead=1, seed=0): ...
def build_dataset(board, source="exact" | "mc", n_positions, seed) -> path
def train(model_name, dataset_path, input_kind, task, seed) -> model_path
```

## Implementation steps

- [x] Dataset builder for both sources. Save to `data/`.
- [x] kNN and sklearn baselines (quick sanity check of learnability).
- [x] From-scratch perceptron and sigmoid unit.
- [x] MLP with early stopping. Produce learning curves (train vs validation error per epoch).
- [x] `LearnedPlayer` and the `search-learned` variant.
- [ ] CNN over the edge grid (optional, `[outside]`).

## How to run

```bash
# datasets (data/ is git-ignored; regenerate with these commands)
python -m solvers.supervised.dataset --board 2x2 --source exact --n 50000 --seed 0
python -m solvers.supervised.dataset --board 2x3 --source exact --n 50000 --seed 0
python -m solvers.supervised.dataset --board 3x3 --source mc --n 50000 --seed 0 --rollouts 200 --workers 4

# one model
python -m solvers.supervised.models --data data/sup_exact_2x3_n50000_s0.npz --model mlp --input both --task reg \
    --seed 0 --params '{"hidden": [64, 64]}'

# play it
python -m harness.cli tournament --board 3x3 --games 200 --seed 0 \
    --players learned:data/models/<model>.pkl,search-learned:data/models/<model>.pkl,greedy,rules

# report experiments
python -m experiments.e4_learned_evaluator --data data/sup_exact_2x3_n50000_s0.npz --seed 0
python -m experiments.e4_bias_variance --data data/sup_exact_2x3_n50000_s0.npz --input raw --seeds 0,1,2
python -m experiments.e5_deductive_vs_inductive --board 3x3 --model data/models/<model>.pkl --games 200 --seed 0
```

## Tests / acceptance

- No game appears in both train and test.
- The MLP beats the linear model on test error (otherwise report it as a finding).
- **% optimal moves** (agreement with solver 2's `best_moves`) on 2×2 / 2×3 test positions is reported for every model.
- The best `LearnedPlayer` beats Greedy on 3×3.

## Report experiments

- E4: test error and % optimal moves per model × input kind.
- Bias–variance: vary MLP hidden size; plot train and test error.
- E5: learned vs rule-based (solver 4).

## Decision log

- **2026-11-12: dedup key is `(canonical(edges), mover_is_first)`.** `V` depends only on `edges`, but `phi` feature 10 also depends on whether the mover is the first player, so the same edges with a different `mover_is_first` is a different input.
- **2026-11-12: split by game, drawn when the game starts.** Each game is assigned to train / val / test (70 / 15 / 15) before it is played; a deduplicated position belongs to the first game it appears in. No game, and no position up to symmetry, is in two splits. Split sizes are therefore only approximately 70 / 15 / 15.
- **2026-11-12: dataset sizes below the 50k target on small boards.** 2×2 has only 583 unique positions (up to symmetry and `mover_is_first`) reachable by the sampling mix, and 2×3 stops at about 33.7k when 500 games in a row add nothing new. 3×3 reaches 50k.
- **2026-11-12: 3×3 uses MC labels only.** Solver 2 solves 2×3 in under a second but 3×3 (24 edges) is not practical in pure Python, so % optimal moves is reported on 2×2 / 2×3 only.
- **2026-11-12: self-play mix and first player.** Each seat is random (0.3), greedy (0.4) or depth-2 search (0.3); the first player is drawn at random per game so `to_move` and `first_player` both vary in the raw input. Depth-2 (no time limit) keeps data generation fast.
- **2026-11-12: one value scale for every model.** Every model outputs a score in `[-1, 1]`: the regression prediction, `2 P(win) - 1` for classifiers, `tanh(w·x)` for the perceptron. `V̂ = score × boxes_remaining` (net boxes, CLAUDE.md §5), used by the one-ply player and as solver 3's evaluator. For MC labels `2p - 1` is a probability, not a box margin, so `V̂` is a surrogate there.
- **2026-11-12: from-scratch learners keep the best-validation weights** (early stopping). The sigmoid unit maps regression targets to `[0, 1]` with `(y + 1) / 2`.
- **2026-11-12: SVM is trained on a seeded 10k subsample** of the training set, because kernel SVMs scale roughly quadratically.
- **2026-11-12: `first_player = 0` at play time.** Like solvers 3 and 4, the player relies on the harness convention that seat 0 moves first.
- **2026-11-12: harness names.** Learned players are entered as `learned:<model_path>` and `search-learned:<model_path>` (CSV name `learned-<model>` / `search-learned`). Two learned players of the same model type in one tournament would share a CSV name.
- **2026-11-12: `MLP(early_stopping=False)`** trains for all epochs and keeps the final weights; used by the unregularised bias–variance runs, where early stopping would hide overfitting.
- **2026-11-12: models are saved with `pickle`** (`data/models/*.pkl`, git-ignored). Only load model files you produced yourself.
