import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import plot_learners_roc
import torch
import random


def fix_seed(seed):
    print(f"Random seed: {seed}\n")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


fix_seed(42)


def main():
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    feature_names = list(train_df.drop(['target'], axis=1).columns)

    # AdaBoost
    clf_adaboost = AdaBoostClassifier(
        input_dim=X_train.shape[1],
        num_learners=10,
        num_epochs=500,
        learning_rate=0.001,
    )
    clf_adaboost.fit(
        X_train,
        y_train
    )
    y_pred_classes, y_pred_probs = clf_adaboost.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=[y_pred_probs],
        y_trues=y_test,
        fpath='./adaboost_roc_curve.png',
    )
    clf_adaboost.compute_feature_importance()
    clf_adaboost.plot_feature_importance(feature_names)

    # Bagging
    clf_bagging = BaggingClassifier(
        input_dim=X_train.shape[1],
    )
    clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=500,
        learning_rate=0.01,
    )
    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=[y_pred_probs],
        y_trues=y_test,
        fpath='./bagging_roc_curve.png',
    )
    clf_bagging.compute_feature_importance()
    clf_bagging.plot_feature(feature_names)

    # Decision Tree
    clf_tree = DecisionTree(
        max_depth=7,
    )
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')
    clf_tree.plot_feature_importance_img(feature_names)


if __name__ == '__main__':
    main()
