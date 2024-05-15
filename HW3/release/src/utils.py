import typing as t
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layer model.
    No non-linear activation allowed.
    """
    
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def entropy_loss(outputs, targets):
    sigmoid_outputs = torch.sigmoid(outputs)  # Sigmoid to convert logits to probabilities
    targets = targets.float()  # Ensure targets are float for computation
    loss = -(targets * torch.log(sigmoid_outputs) + (1 - targets) * torch.log(1 - sigmoid_outputs))
    return torch.mean(loss)  # Return mean of losses


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    plt.figure()
    for idx, y_pred in enumerate(y_preds):
        fpr, tpr, _ = roc_curve(y_trues, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Learner {idx+1} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal dashed line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(fpath)  # Save the figure to file
    plt.close()  # Close the plot to free up memory

def plot_feature_importance_img(feature_importances, column_names, output_path='feature_importance.png'):
    if feature_importances is None:
        raise ValueError("Feature importances have not been computed. Call the fit method first.")

    # Sorting feature importances
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_importances = np.array(feature_importances)[sorted_indices]
    sorted_columns = np.array(column_names)[sorted_indices]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_columns, sorted_importances, align='center')
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.savefig(output_path)
    plt.show()    
