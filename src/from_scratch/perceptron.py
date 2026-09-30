"""Perceptron, Delta rule (Adaline / LMS) and One-vs-Rest — CO3117 Ch.3, Depth A.

Conventions (the tests and experiment scripts rely on them):
- Binary labels y are in {-1, +1}.
- Weights start at zero unless stated otherwise; bias b is stored separately from w.
- Samples are visited in the given order unless shuffle=True (then use np.random.default_rng(seed)).
"""
import numpy as np


class Perceptron:
    """Rosenblatt perceptron for binary labels in {-1, +1}.

    Prediction: y_hat = sign(w.x + b), with sign(0) = +1.
    Update (only when sample i is misclassified, i.e. y_i * (w.x_i + b) <= 0):
        w <- w + lr * y_i * x_i
        b <- b + lr * y_i

    Attributes set by fit():
        w_        (d,) weight vector
        b_        float bias
        errors_   list[int]  number of updates (mistakes) in each epoch
        history_  list[(w copy, b)]  state after EVERY update (for the geometry plot)
    Training stops early when an epoch has zero mistakes.
    """

    def __init__(self, lr=1.0, n_epochs=50, shuffle=False, seed=42):
        self.lr = lr
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        self.errors_ = []
        self.history_ = []
        rng = np.random.default_rng(self.seed)

        for _ in range(self.n_epochs):
            order = rng.permutation(n) if self.shuffle else np.arange(n)
            mistakes = 0
            for i in order:
                # Misclassified (or exactly on the boundary): y_i * (w.x_i + b) <= 0
                if y[i] * (X[i] @ self.w_ + self.b_) <= 0:
                    self.w_ = self.w_ + self.lr * y[i] * X[i]
                    self.b_ = self.b_ + self.lr * y[i]
                    self.history_.append((self.w_.copy(), self.b_))
                    mistakes += 1
            self.errors_.append(mistakes)
            if mistakes == 0:  # converged: a full clean pass
                break
        return self

    def decision_function(self, X):
        """Return w.x + b for every row of X, shape (n,)."""
        return np.asarray(X, dtype=float) @ self.w_ + self.b_

    def predict(self, X):
        """Return labels in {-1, +1}; sign(0) = +1."""
        return np.where(self.decision_function(X) >= 0, 1, -1)


class DeltaRule:
    """Delta rule / Adaline / LMS: linear output o = w.x + b (NO threshold during training).

    Minimises E = 1/2 * mean_i (y_i - o_i)^2 by batch gradient descent:
        w <- w + lr * mean_i[(y_i - o_i) * x_i]
        b <- b + lr * mean_i[(y_i - o_i)]
    Prediction: sign(o), with sign(0) = +1.

    Attributes set by fit():
        w_, b_
        losses_   list[float]  value of E after each epoch
    """

    def __init__(self, lr=0.01, n_epochs=100):
        self.lr = lr
        self.n_epochs = n_epochs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        self.losses_ = []

        for _ in range(self.n_epochs):
            o = X @ self.w_ + self.b_          # linear output, no threshold
            err = y - o                        # (t - o)
            self.w_ = self.w_ + self.lr * (X.T @ err) / n
            self.b_ = self.b_ + self.lr * err.mean()
            o_new = X @ self.w_ + self.b_
            self.losses_.append(0.5 * np.mean((y - o_new) ** 2))
        return self

    def decision_function(self, X):
        """Return the linear output o = w.x + b, shape (n,)."""
        return np.asarray(X, dtype=float) @ self.w_ + self.b_

    def predict(self, X):
        """Return labels in {-1, +1}; sign(0) = +1."""
        return np.where(self.decision_function(X) >= 0, 1, -1)


class OneVsRest:
    """Multi-class wrapper: one binary model per class (class k -> +1, others -> -1).

    make_model: a function with no arguments that returns a fresh binary model,
                e.g. OneVsRest(lambda: Perceptron(n_epochs=20)).
    predict(X) returns the class whose model gives the LARGEST decision_function value.

    Attributes set by fit():
        classes_  sorted unique labels of y
        models_   list of fitted binary models, same order as classes_
    """

    def __init__(self, make_model):
        self.make_model = make_model

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)  # sorted
        self.models_ = []
        for k in self.classes_:
            y_bin = np.where(y == k, 1, -1)
            self.models_.append(self.make_model().fit(X, y_bin))
        return self

    def predict(self, X):
        scores = np.column_stack([m.decision_function(X) for m in self.models_])
        return self.classes_[np.argmax(scores, axis=1)]