import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001):
        """Implement your code here"""
        losses_of_models = []
        for model in self.learners:
            ...

        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        raise NotImplementedError

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""
        raise NotImplementedError
