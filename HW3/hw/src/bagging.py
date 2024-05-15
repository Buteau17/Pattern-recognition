import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
from .utils import WeakClassifier

class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        # Create 10 learners
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]
        self.feature_importances= None

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """Train each weak classifier on a bootstrap sample of the training data"""
        losses_of_models = []
        criterion = nn.BCELoss()
        

        for model in self.learners:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            model_losses = []
            for epoch in range(num_epochs):
                # Create a bootstrap sample
                indices = np.random.choice(len(X_train), len(X_train), replace=True)
                X_bootstrap = X_train[indices]
                y_bootstrap = y_train[indices]

                X_bootstrap = torch.tensor(X_bootstrap, dtype=torch.float32)
                y_bootstrap = torch.tensor(y_bootstrap, dtype=torch.float32).view(-1, 1)

                # Forward pass
                outputs = model(X_bootstrap)
                outputs = torch.sigmoid(outputs)  # Apply sigmoid activation

                loss = criterion(outputs, y_bootstrap)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                model_losses.append(loss.item())

            losses_of_models.append(model_losses)

        return losses_of_models

    def predict_learners(self, X) -> t.Tuple[t.Sequence[int], t.Sequence[float]]:
        """Aggregate predictions from all learners and return the final prediction"""
        X = torch.tensor(X, dtype=torch.float32)
        all_preds = []

        for model in self.learners:
            with torch.no_grad():
                preds = torch.sigmoid(model(X)).numpy()  # Apply sigmoid activation
                all_preds.append(preds)

        all_preds = np.array(all_preds).squeeze()
        avg_preds = np.mean(all_preds, axis=0)
        binary_preds = (avg_preds >= 0.5).astype(int)
        return binary_preds, avg_preds

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Compute feature importances by averaging the absolute weights of all learners"""
        importances = np.zeros_like(self.learners[0].linear.weight.data.numpy())
        for learner in self.learners:
            importances += np.abs(learner.linear.weight.data.numpy())
        importances /= len(self.learners)
        self.feature_importances = importances.sum(axis=0).tolist()
        return self.feature_importances

    def plot_feature(self, column_names):
        if self.feature_importances is None:
            raise ValueError("Feature importances have not been computed. Call the fit method first.")

        # Ensure feature_importances_ is a numpy array for proper indexing
        importances_array = np.array(self.feature_importances)

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
        plt.savefig('feature2.png')
        # plt.show()
