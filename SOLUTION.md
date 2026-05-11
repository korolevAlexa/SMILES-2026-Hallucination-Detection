## Solution Report

I started from the baseline pipeline. In the original `aggregation.py`, each example was represented by the hidden state of the last real token from the final transformer layer. With the original `probe.py`, this gave about:

```text
Accuracy: 73.08%
F1:       81.82%
AUROC:    73.80%
```

The dataset is imbalanced: there are 483 hallucinated answers and 206 truthful answers. Because of this, the majority-class baseline already reaches about 70% accuracy. Therefore, during development I looked not only at accuracy, but also at F1 and AUROC.

The first improvement was in `aggregation.py`. Instead of using only the final layer, I averaged the final-token representations over the last four transformer layers:

```python
real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
last_pos = int(real_positions[-1].item())

last_token_layers = hidden_states[-4:, last_pos, :]
feature = last_token_layers.mean(dim=0)
```

This improved internal test AUROC to about 74.9%. I also tried concatenating several layer representations, but this increased feature dimensionality too much and performed worse. This suggested that the aggregation should stay compact.

I then performed exploratory data analysis. The most useful observation was that response length is strongly related to the label. Truthful answers were noticeably shorter: their median length was about 34.5 words, while hallucinated answers had a median length of about 77 words. Hallucinated answers were also more likely to reach the 512-token limit and be truncated.

Based on this, I added two scalar length features in `aggregation.py`: normalized sequence length and a truncation flag.

```python
n_real = float(real_positions.numel())
max_length = float(MAX_SEQUENCE_LENGTH)

scalar_features = torch.stack(
    [
        feature.new_tensor(n_real / max(max_length, 1.0)),
        feature.new_tensor(float(n_real >= max_length)),
    ]
)

return torch.cat([feature, scalar_features], dim=0)
```

The final feature dimension became 898: 896 hidden-state features and 2 scalar length features. A length-only baseline reached approximately 0.69 AUROC, which indicates that response length contains a meaningful signal, although it is not sufficient as a standalone representation. In contrast, a TF-IDF text-only baseline achieved only about 0.52 AUROC. This suggests that simple lexical statistics are not enough for this task, and that contextual hidden-state representations are more informative.

I also evaluated several geometric and topological feature sets, as suggested in the task description. These included layer-wise activation norms, cosine similarities between layer representations, inter-layer representation drift, token-level drift over the tail of the sequence, pairwise distances between token representations, and SVD-based spectral features. However, these features did not lead to a consistent improvement. A lightweight geometric variant reached approximately 75.0% AUROC, while a heavier version increased the feature dimension to 944 and reduced internal test AUROC to about 71.8%. This suggests that the additional geometric statistics introduced noise and increased variance on the small dataset. Therefore, the final aggregation keeps only the more stable representation: averaged final-layer hidden states and two length-based scalar features.

The next part of the solution focused on `probe.py`. The original probe was a one-hidden-layer MLP with 256 hidden units trained with binary cross-entropy. During experiments, this model frequently achieved near-perfect train AUROC, while validation and test AUROC remained substantially lower. This behavior indicated overfitting: the probe had enough capacity to fit the training split, but did not generalize well to unseen examples.

To make the training objective better aligned with ranking quality, I introduced a pairwise ranking loss. The goal of this loss is to assign higher scores to hallucinated examples than to truthful examples:

```python
pos_logits = logits[pos_mask]
neg_logits = logits[neg_mask]

pairwise_diff = pos_logits[:, None] - neg_logits[None, :]
ranking_loss = torch.nn.functional.softplus(-pairwise_diff).mean()

bce_loss = criterion(logits, y_t)
loss = 0.7 * ranking_loss + 0.3 * bce_loss
```

The BCE component is kept as a stabilizing term, while the ranking component directly encourages better separation between positive and negative examples by score. This was useful because AUROC depends on the relative ordering of scores rather than only on a fixed classification threshold.

I also reduced the capacity of the probe. A sweep over hidden sizes showed that a smaller head generalized better than the original 256-unit MLP. The final probe uses 48 hidden units and dropout:

```python
self._net = nn.Sequential(
    nn.Linear(input_dim, 48),
    nn.ReLU(),
    nn.Dropout(0.10),
    nn.Linear(48, 1),
)
```

This architecture provides a better bias-variance trade-off for the small dataset: it remains expressive enough to use the hidden-state features, but is less prone to memorizing the training split.

I additionally added threshold tuning inside `fit`. This is important because the final `predictions.csv` contains binary labels rather than probabilities. Instead of using a fixed threshold of 0.5, the probe searches over candidate thresholds and selects one based on accuracy, balanced accuracy, and predicted class balance, while ignoring thresholds that collapse to a single predicted class. This is especially relevant because the dataset is imbalanced. It is also necessary because the final probe in `solution.py` is trained directly before prediction and does not receive a separate validation-thresholding step.

I also tested PCA-based dimensionality reduction in `probe.py`. PCA followed by a small MLP produced results close to the final model, but did not provide a stable improvement. Since it also made the pipeline more complex, PCA was not included in the final solution. This suggests that the smaller MLP already provides sufficient regularization for the hidden-state feature space.

Finally, I evaluated several splitting strategies: repeated holdout, label-only stratification, label-plus-response-length stratification, and k-fold splitting. The results were sensitive to the exact split, which is expected given the small dataset size. Length-aware stratification sometimes improved the average holdout result, but the improvement was not consistent. K-fold evaluation produced a more conservative estimate, around 71–72.5% AUROC. Since these alternatives did not provide a clear and stable advantage, I kept the default single stratified split in the final version.

The final solution uses:

- `aggregation.py`: average of the final-token representation over the last four transformer layers, plus two length features;
- `probe.py`: small MLP with 48 hidden units, dropout 0.10, mixed ranking/BCE loss, and threshold tuning;
- `splitting.py`: default single stratified split.

The final full run gave approximately:

```text
Accuracy: 75.96%
F1:       84.28%
AUROC:    75.52%
```

Compared with the original baseline, internal test AUROC improved from about 73.8% to about 75.5%. The final solution was selected because it provides a compact and interpretable feature representation, reduces overfitting in the probe, and performed more reliably than the more complex geometric or PCA-based alternatives.
