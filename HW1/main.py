import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None
        

        self.epoch_list = []
        self.training_loss_list = []

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
       X= np.insert(X, 0, 1, axis=1)
       X_transpose = np.transpose(X)
       self.weights = np.dot(np.dot(np.linalg.inv(np.dot(X_transpose, X)) , X_transpose), y)
       self.intercept = self.weights[0]
       self.weights = self.weights[1:]

    def predict(self, X):
        return np.dot(X, self.weights) + self.intercept


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        X= np.insert(X, 0, 1, axis=1)
        y = np.squeeze(y)
        num_samples , num_features = X.shape
        self.weights = np.zeros (num_features)
         #gradient descent

        for epoch in range (1, epochs):
            y_pred = np.dot (X, self.weights.T)
            error = y_pred - y
            if epoch % 10000 == 0:
                self.epoch_list.append(epoch)
                self.training_loss_list.append(compute_mse(y_pred, y))
                # print(np.mean(error))
            gradient = np.dot(X.T, error) / num_samples
            # L1 regularization
            #gradient = np.dot(X.T, error) / num_samples + (1/num_samples) * np.abs(self.weights)
            self.weights -= learning_rate * gradient
        self.intercept=  self.weights[0]
        self.weights =  self.weights[1:]    

        
    def predict(self, X):
        return np.dot(X, self.weights) + self.intercept

    def plot_learning_curve(self):
        plt.plot(self.epoch_list, self.training_loss_list)
        plt.title("Training Loss")
        plt.xlabel("epoch"), plt.ylabel("MSE loss")
        plt.savefig('output.png')


def compute_mse(prediction, ground_truth):
    mse= np.mean(( prediction - ground_truth ) **2 )
    return mse


def main():
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()
    print(train_x.shape)
    print(type(train_x))

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    LR_GD.fit(train_x, train_y, learning_rate=0.0001, epochs=2000000)
    LR_GD.plot_learning_curve()
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).sum()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = ((mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
