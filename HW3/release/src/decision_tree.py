import numpy as np
import matplotlib.pyplot as plt

"""
You dont have to follow the stucture of the sample code.
However, you should checkout if your class/function meet the requirements.
"""



class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.feature_importances_ = None

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1])
        self.total_samples = X.shape[0] 
        self.tree = self._grow_tree(X, y)
        
        

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if depth < self.max_depth and num_samples > 1:
            best_split = find_best_split(X, y)
           
            
            if best_split:
                self.feature_importances_[best_split['feature_index']] +=1
                left_indices, right_indices = split_dataset(X, y, best_split['feature_index'], best_split['threshold'])
                left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
                right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
                return {'feature_index': best_split['feature_index'], 'threshold': best_split['threshold'],
                        'left': left_subtree, 'right': right_subtree}
        return {'leaf': True, 'value': np.mean(y)}

    def predict(self, X, threshold=0.5):
        continuous_preds = np.array([self._predict_tree(x, self.tree) for x in X])
        binary_preds = (continuous_preds >= threshold).astype(int)
        return binary_preds

    def _predict_tree(self, x, tree_node):
        if tree_node.get('leaf'):
            return tree_node['value']
        feature_index = tree_node['feature_index']
        threshold = tree_node['threshold']
        if x[feature_index] <= threshold:
            return self._predict_tree(x, tree_node['left'])
        else:
            return self._predict_tree(x, tree_node['right'])


    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, column_names):
        if self.feature_importances_ is None:
            raise ValueError("Feature importances have not been computed. Call the fit method first.")

        # Sorting feature importances
        sorted_indices = np.argsort(self.feature_importances_)[::-1]
        sorted_importances = self.feature_importances_[sorted_indices]
        sorted_columns = np.array(column_names)[sorted_indices]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_columns, sorted_importances, align='center')
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.savefig('feature3.png')
        # plt.show()
  



    


def split_dataset(X, y, feature_index, threshold):
    left_indices = np.where(X[:, feature_index] <= threshold)[0]
    right_indices = np.where(X[:, feature_index] > threshold)[0]
    return left_indices, right_indices


def find_best_split(X, y):
    num_samples, num_features = X.shape
    if num_samples <= 1:
        return None

    best_split = {}
    best_gain = -1

    for feature_index in range(num_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_indices, right_indices = split_dataset(X, y, feature_index, threshold)
            if len(left_indices) == 0 or len(right_indices) == 0:
                continue

            gain = information_gain(y, y[left_indices], y[right_indices])
            if gain > best_gain:
                best_gain = gain
                best_split = {
                    'feature_index': feature_index,
                    'threshold': threshold
                }

    return best_split if best_gain > 0 else None


def gini_index(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return 1 - np.sum([p ** 2 for p in ps if p > 0])

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

y = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
print("Gini Index of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]:", gini_index(y))
print("Entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]:", entropy(y))


def information_gain(parent, left_child, right_child):
    num_left = len(left_child)
    num_right = len(right_child)
    num_total = len(parent)

    p_left = num_left / num_total
    p_right = num_right / num_total

    return entropy(parent) - (p_left * entropy(left_child) + p_right * entropy(right_child))
