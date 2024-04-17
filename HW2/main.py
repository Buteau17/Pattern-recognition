import typing as t
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        n_samples, n_features = inputs.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0


        for _ in range(self.num_iterations):
            linear_model = np.dot(inputs, self.weights) + self.intercept
            y_pred = self.sigmoid(linear_model)
            # if _ % 50000 == 0:
            #     print(np.mean(y_pred - y))

            dw = (1 / n_samples) * np.dot(inputs.T, (y_pred - targets))
            db = (1 / n_samples) * np.sum(y_pred - targets)

            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        linear_model = np.dot(inputs, self.weights) + self.intercept
        y_pred = self.sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred, y_pred_class

    
    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1 /(1 + np.exp(-x))


class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        X0 = inputs[targets==0]
        X1 = inputs[targets==1]
        self.m0 = np.mean(X0, axis=0)
        self.m1 = np.mean(X1, axis=0)
        self.sw = np.dot((X0 - self.m0).T, (X0 - self.m0)) + np.dot((X1 - self.m1).T, (X1 - self.m1))
        
        self.sb = np.outer(self.m1 - self.m0, self.m1 - self.m0)
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(self.sw).dot(self.sb))
        self.w = eigvecs[:, np.argmax(eigvals)]

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Sequence[t.Union[int, bool]]:
        projected_data = np.dot(inputs, self.w)
        projected_mean_0 = np.dot(self.m0, self.w)
        projected_mean_1 = np.dot(self.m1, self.w)

        y_pred = []
        for data_point in projected_data:
            distance_to_mean_0 = abs(data_point - projected_mean_0)
            distance_to_mean_1 = abs(data_point - projected_mean_1)

            if distance_to_mean_0 < distance_to_mean_1:
                y_pred.append(0)
            else:
                y_pred.append(1)

        return np.array(y_pred)
    def plot_projection(self, inputs: npt.NDArray[float], file_name : str):
        fig, ax = plt.subplots(figsize=(10,10))
        colors=['red','blue']
        y_pred = self.predict(inputs)
        slope = self.w[1]/self.w[0] # compute slope
        x = np.linspace(-0.5,1.5,200)
        y = slope*x+0
        plt.plot(x,y, '-g')
        
        for point, pred in zip(inputs, y_pred):
            point_project = (np.dot(point,self.w)*self.w) / np.dot(self.w.T, self.w) # project point
            # draw points and lines
            plt.title(f"Projection Line: w={slope}, b=0")
            plt.plot( [point_project[0], point[0]],[point_project[1], point[1]], color='black', linewidth=0.1,zorder=1) 
            plt.scatter(point_project[0], point_project[1], color=colors[pred], s=5, zorder=2)
            plt.scatter(point[0],point[1],color=colors[pred], s=5, zorder=3)
        plt.savefig(f'{file_name}.png')

def compute_auc(y_trues, y_preds) -> float:
    return roc_auc_score(y_trues, y_preds)

def accuracy_score(y_trues, y_preds) -> float:
    return np.mean(np.array(y_trues) == np.array(y_preds))


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=0.00001,  # You can modify the parameters as you want
        num_iterations=3000000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    FLD_.fit(x_train, y_train)
    y_preds = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_preds)
    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')
    FLD_.plot_projection(x_test, 'test')


if __name__ == '__main__':
    main()
