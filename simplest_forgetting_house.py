import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_ITER = 150
LR = 1e-7


def loss(h, y):
    sq_error = (h - y) ** 2
    n = len(y)
    return 1.0 / (2 * n) * sq_error.sum()


class LinearRegression:

    def __init__(self):

        self._W = np.zeros(2)
        self._cost_history = []
        self._w_history = [self._W]

    def predict(self, X):

        return np.dot(X, self._W)

    def _gradient_descent_step(self, X, targets, lr):

        predictions = self.predict(X)

        error = predictions - targets
        gradient = np.dot(X.T, error) / len(X)

        self._W -= lr * gradient

    def fit(self, X, y, n_iter=100000, lr=0.01):

        for _ in range(n_iter):

            prediction = self.predict(X)
            cost = loss(prediction, y)

            self._cost_history.append(cost)

            self._gradient_descent_step(X, y, lr)

            self._w_history.append(self._W.copy())

        return self


def plot_cost(splt, title, cost_history):
    splt.set_title(f"Cost Function J - {title}")
    splt.set_xlabel("No. of iterations")
    splt.set_ylabel("Cost")
    splt.plot(cost_history)


def plot_line_scatter(splt, title, x, y, model, scolor, lcolor):
    splt.set_title("House Prices vs. Living Area")
    splt.set_xlabel("GrLivArea")
    splt.set_ylabel("SalePrice")
    splt.set_ylim(0, 800000)

    """ scatter plot """
    plt.scatter(x[:, 1], y, s=32, c=scolor)

    """ line"""
    plt_x = np.linspace(0, 7000, 1000)
    plt_y_cumul = model._W[1] * plt_x + model._W[0]
    plt.plot(
        plt_x,
        plt_y_cumul,
        color=lcolor,
        label=f"{title}: y={model._W[1]:.2f}x+{model._W[0]:.2f}",
    )
    plt.legend()


if __name__ == "__main__":

    df_train = pd.read_csv("dataset/house_prices_train.csv")

    """ CUMULATIVE TRAINING"""

    plt.figure(figsize=(15, 10))

    cumul_x = df_train["GrLivArea"]
    cumul_y = df_train["SalePrice"]

    #  for bias
    cumul_x = np.c_[np.ones(cumul_x.shape[0]), cumul_x]

    # train
    cumul_clf = LinearRegression()
    cumul_clf.fit(cumul_x, cumul_y, n_iter=N_ITER, lr=LR)

    parameters = "Weight: %.2f  Bias: %.2f" % (cumul_clf._W[1], cumul_clf._W[0])
    print(f"cumulative paarameters: \n {parameters}")

    plot_line_scatter(
        plt.subplot(2, 2, 1), "Cumulative", cumul_x, cumul_y, cumul_clf, "cyan", "green"
    )
    plot_cost(plt.subplot(2, 2, 2), "Cumulative", cumul_clf._cost_history)

    """ CONTINUAL LEARNING """

    df_new = df_train[df_train.YearBuilt > 2000]
    df_old = df_train[df_train.YearBuilt <= 2000]

    """ first experience """
    x_old = df_old["GrLivArea"]
    y_old = df_old["SalePrice"]

    x_old = np.c_[np.ones(x_old.shape[0]), x_old]

    cl_clf = LinearRegression()
    cl_clf.fit(x_old, y_old, n_iter=150, lr=1e-7)

    old_parameters = "Weight: %.2f  Bias: %.2f" % (cl_clf._W[1], cl_clf._W[0])
    print(f"paarameters after first experience: \n {old_parameters}")

    plot_line_scatter(
        plt.subplot(2, 2, 3),
        "Cumulative",
        cumul_x,
        cumul_y,
        cumul_clf,
        "white",
        "green",
    )
    plot_line_scatter(
        plt.subplot(2, 2, 3), "Old", x_old, y_old, cl_clf, "gray", "black"
    )

    """ second experience """

    x_new = df_new["GrLivArea"]
    y_new = df_new["SalePrice"]

    x_new = np.c_[np.ones(x_new.shape[0]), x_new]

    cl_clf.fit(x_new, y_new, n_iter=150, lr=1e-7)

    new_parameters = "Weight: %.2f  Bias: %.2f" % (cl_clf._W[1], cl_clf._W[0])
    print(f"paarameters after second experience: \n {new_parameters}")

    plot_line_scatter(
        plt.subplot(2, 2, 3), "New", x_new, y_new, cl_clf, "orange", "red"
    )
    plot_cost(
        plt.subplot(2, 2, 4),
        "Cost Function J - Continual Learning",
        cl_clf._cost_history,
    )

    plt.tight_layout()
    plt.show()
