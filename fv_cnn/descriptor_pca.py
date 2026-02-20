from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA


class LocalPCATransform:
    def __init__(
        self,
        num_components: int,
        whitening: bool = False,
        whitening_regul: float = 0.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.num_components = int(num_components)
        self.whitening = bool(whitening)
        self.whitening_regul = float(whitening_regul)
        self.random_state = random_state
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "LocalPCATransform":
        if X.ndim != 2:
            raise ValueError("X must be 2D (N, D)")
        if self.num_components <= 0:
            raise ValueError("num_components must be > 0")
        if self.num_components > X.shape[1]:
            raise ValueError("num_components cannot exceed descriptor dimension")

        pca = PCA(
            n_components=self.num_components,
            svd_solver="randomized",
            random_state=self.random_state,
        )
        pca.fit(X)
        self.mean_ = pca.mean_.astype(np.float32)
        self.components_ = pca.components_.astype(np.float32)
        self.explained_variance_ = pca.explained_variance_.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None or self.explained_variance_ is None:
            raise RuntimeError("LocalPCATransform is not fitted")
        Xc = X - self.mean_
        proj = np.dot(Xc, self.components_.T)
        if self.whitening:
            denom = np.sqrt(self.explained_variance_ + self.whitening_regul)
            proj = proj / denom
        return proj.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
