import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model = []
        self.alphas = []

    def fit(self, X, y):
        n = X.shape[0]
        weights = np.ones(n) / n

        for _ in range(self.n_estimators):
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=weights)
            predictions = stump.predict(X)

            incorrect = predictions != y
            error = np.dot(weights, incorrect)
            error = max(error, 1e-10)  # avoid divide-by-zero

            alpha = 0.5 * np.log((1 - error) / error)
            alpha *= self.learning_rate

            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            self.model.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        final = np.zeros(X.shape[0])
        for model, alpha in zip(self.model, self.alphas):
            final += alpha * model.predict(X)
        return np.sign(final).astype(int)
