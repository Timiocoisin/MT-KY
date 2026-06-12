"""HOG + LogisticRegression baseline (sklearn, non-differentiable)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.linear_model import Ridge


@dataclass
class HogLRModel:
    orientations: int = 9
    pixels_per_cell: tuple[int, int] = (16, 16)
    cells_per_block: tuple[int, int] = (2, 2)
    regressor: Ridge | None = None

    def _extract(self, image: np.ndarray) -> np.ndarray:
        gray = rgb2gray(image)
        return hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            feature_vector=True,
        )

    def fit(self, images: list[np.ndarray], wear_pct: np.ndarray) -> None:
        feats = np.stack([self._extract(img) for img in images], axis=0)
        self.regressor = Ridge(alpha=1.0)
        self.regressor.fit(feats, wear_pct)

    def predict(self, images: list[np.ndarray]) -> np.ndarray:
        if self.regressor is None:
            raise RuntimeError("Model not fitted")
        feats = np.stack([self._extract(img) for img in images], axis=0)
        return self.regressor.predict(feats).clip(0.0, 100.0)
