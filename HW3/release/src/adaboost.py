import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier
import matplotlib.pyplot as plt
from loguru import logger

class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10, num_epochs: int = 500, learning_rate: float = 0.001) -> None:
        self.input_dim = input_dim
        self.num_learners = num_learners
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.learners = []
        self.alphas = []
        self.feature_importances_ = None

    def fit(self, X_train, y_train):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        n_samples = X_train.shape[0]
        w = np.full(n_samples, 1 / n_samples)
        epsilon = 1e-10  # Small value to avoid division by zero

        for _ in range(self.num_learners):
            sample_indices = np.random.choice(np.arange(n_samples), size=n_samples, p=w)
            X_sampled = X_train[sample_indices]
            y_sampled = y_train[sample_indices]

            model = WeakClassifier(input_dim=self.input_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            for _ in range(self.num_epochs):
                for i in range(n_samples):
                    X_sample = X_sampled[i:i+1]
                    y_sample = y_sampled[i:i+1]
                    weight = w[i]

                    model.train()
                    optimizer.zero_grad()
                    output = model(X_sample)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, y_sample)
                    weighted_loss = weight * loss
                    weighted_loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                preds = torch.sigmoid(model(X_train)) > 0.5
                missclassified = preds.numpy().flatten() != y_train.numpy().flatten()
                error = np.dot(w, missclassified) / np.sum(w)

            if error == 0:
                alpha = float('inf')
            else:
                error = np.clip(error, epsilon, 1 - epsilon)
                alpha = 0.5 * np.log((1 - error) / error)

            w *= np.exp(-alpha * (missclassified * 2 - 1))
            w = np.clip(w, epsilon, 1 - epsilon)
            w /= np.sum(w)

            self.learners.append(model)
            self.alphas.append(alpha)

            logger.debug(f'Learner trained with alpha: {alpha}, final error: {error}')

        logger.debug(f'Total alphas: {len(self.alphas)}')

    def predict(self, X) -> t.Tuple[np.ndarray, np.ndarray]:
        X = torch.tensor(X, dtype=torch.float32)
        learner_predictions = []
        learner_probabilities = []

        for learner, alpha in zip(self.learners, self.alphas):
            with torch.no_grad():
                outputs = learner(X)
                probs = torch.sigmoid(outputs).squeeze().numpy()
                preds = (probs > 0.5).astype(int) * 2 - 1  # Convert to -1, 1 for weighted sum
                learner_predictions.append(alpha * preds)
                learner_probabilities.append(probs)

        learner_predictions = np.array(learner_predictions)  # Shape: (num_learners, num_samples)
        learner_probabilities = np.array(learner_probabilities)  # Shape: (num_learners, num_samples)
        alphas = np.array(self.alphas)  # Shape: (num_learners,)

        logger.debug(f'Learner predictions shape: {learner_predictions.shape}')
        logger.debug(f'Alphas shape: {alphas.shape}')

        weighted_votes = np.sum(learner_predictions, axis=0)
        final_predictions = (weighted_votes > 0).astype(int)
        final_probabilities = np.average(learner_probabilities, axis=0, weights=alphas)

        return final_predictions, final_probabilities

    def compute_feature_importance(self) -> t.Sequence[float]:
        # Initialize importances to zeros with the same shape as the first layer's weights
        importances = np.zeros_like(self.learners[0].linear.weight.data.numpy())
        total_alpha = sum(self.alphas)

        for learner, alpha in zip(self.learners, self.alphas):
            # Sum the absolute values of the weights scaled by the corresponding alpha
            importances += np.abs(learner.linear.weight.data.numpy()) * (alpha / total_alpha)

        self.feature_importances_ = importances.sum(axis=0).tolist()
        return self.feature_importances_

        
    def plot_feature_importance(self, column_names):
        if self.feature_importances_ is None:
            raise ValueError("Feature importances have not been computed. Call the fit method first.")

        # Ensure feature_importances_ is a numpy array for proper indexing
        importances_array = np.array(self.feature_importances_)

        # Sorting feature importances
        sorted_indices = np.argsort(importances_array)[::-1]
        sorted_importances = importances_array[sorted_indices]
        sorted_columns = np.array(column_names)[sorted_indices]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_columns, sorted_importances, align='center')
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.savefig('feature1.png')
        # plt.show()