"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module``. The classifier uses a small single-hidden-layer
    MLP with ``StandardScaler`` pre-processing. The network is built lazily in
    ``fit()`` once the feature dimension is known.
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None
        self._scaler = StandardScaler()
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Args:
            input_dim: Feature vector dimensionality.
        """
        self._net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(48, 1),
        )

    @staticmethod
    def _choose_threshold(probs: np.ndarray, y: np.ndarray) -> float:
        """Choose a decision threshold, avoiding all-one/all-zero collapse.

        Args:
            probs: Estimated probabilities for the hallucinated class.
            y: Integer labels with values in ``{0, 1}``.

        Returns:
            Decision threshold for converting probabilities to labels.
        """
        probs = np.asarray(probs, dtype=np.float64)
        y = np.asarray(y, dtype=int)

        unique_probs = np.unique(probs)
        if len(unique_probs) > 1:
            midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
            candidates = np.concatenate([midpoints, np.linspace(0.01, 0.99, 99)])
        else:
            candidates = np.linspace(0.01, 0.99, 99)

        best_threshold = 0.5
        best_score = (-1.0, -1.0, -1.0)

        for threshold in np.unique(candidates):
            y_pred = (probs >= threshold).astype(int)

            if y_pred.min() == y_pred.max():
                continue

            acc = accuracy_score(y, y_pred)
            balanced_acc = balanced_accuracy_score(y, y_pred)
            class_balance = -abs(float(y_pred.mean()) - float(y.mean()))
            score = (acc, balanced_acc, class_balance)

            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        return best_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw logits.
        """
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )

        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features with ``StandardScaler``, builds the network if needed,
        and optimises with a mixture of pairwise ranking loss and BCE loss.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self``.
        """
        torch.manual_seed(42)
        np.random.seed(42)

        X_scaled = self._scaler.fit_transform(X)
        self._build_network(X_scaled.shape[1])

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        pos_mask = y_t == 1.0
        neg_mask = y_t == 0.0

        self.train()
        for _ in range(200):
            optimizer.zero_grad()

            logits = self(X_t)

            bce_loss = criterion(logits, y_t)
            pos_logits = logits[pos_mask]
            neg_logits = logits[neg_mask]

            if pos_logits.numel() and neg_logits.numel():
                pairwise_diff = pos_logits[:, None] - neg_logits[None, :]
                ranking_loss = torch.nn.functional.softplus(-pairwise_diff).mean()
                loss = 0.7 * ranking_loss + 0.3 * bce_loss
            else:
                loss = bce_loss

            loss.backward()
            optimizer.step()

        self.eval()

        with torch.no_grad():
            train_probs = torch.sigmoid(self(X_t)).numpy()

        self._threshold = self._choose_threshold(train_probs, y)
        return self

    def fit_hyperparameters(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``.

        Returns:
            ``self``.
        """
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = self._choose_threshold(probs, y_val)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class.
        """
        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()

        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()

        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
