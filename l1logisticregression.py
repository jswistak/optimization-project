import numpy as np
from scipy.optimize import minimize
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class L1LogisticRegression:
    def __init__(self, C: int = 100, method: str = "L-BFGS-B"):
        """
        Initializes the L1LogisticRegression model.

        Args:
            C (int, optional): Inverse of regularization strength; must be a positive float.
            Smaller values specify stronger regularization. Defaults to 100.
            method (str, optional): Optimization algorithm.
            For L1 regularization, 'L-BFGS-B' is typically preferred. Defaults to "L-BFGS-B".
        """
        self.C = C
        self.method = method
        self.coef_ = None
        self.loss_history = []
        self.logloss_history = []

    @staticmethod
    def _reparametrization(X):
        X = np.asarray(X)
        X_plus = np.maximum(X, 0)
        X_minus = np.maximum(-X, 0)

        return X_plus, X_minus

    def _objective_and_grad(self, u, X, y):
        """
        Returns (objective, gradient) for LBFGS-B in the
        (x⁺, x⁻) ≥ 0 re-parametrisation.
        """
        n = X.shape[1]
        x_plus, x_minus = u[:n], u[n:]
        x = x_plus - x_minus

        # forward pass
        z = y * (X @ x)
        log_terms = np.logaddexp(0.0, -z)
        obj = self.C * np.sum(log_terms) + np.sum(u)

        g_w = self.C * (X.T @ (-y / (1.0 + np.exp(z))))

        grad = np.concatenate([g_w + 1.0, -g_w + 1.0])
        return obj, grad

    def _objective(self, u, X, y):
        """
        Calculates the objective function for L1-regularized logistic regression.

        The objective function consists of the logistic loss and the L1 regularization term.
        The logistic loss is computed using log-sum-exp trick for numerical stability.
        The L1 regularization is the sum of absolute values of the coefficients,
        which is implemented by reparametrizing each coefficient x as the difference
        of two non-negative variables x_plus and x_minus, i.e., x = x_plus - x_minus,
        and penalizing the sum of x_plus and x_minus.

        Args:
            u (np.ndarray): A numpy array containing the concatenated x_plus and x_minus variables.
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The target vector.

        Returns:
            float: The value of the objective function.
        """
        X_plus, _ = L1LogisticRegression._reparametrization(X)
        n = X_plus.shape[1]
        x_plus = u[:n]
        x_minus = u[n:]
        x = x_plus - x_minus
        z = y * (X.dot(x))
        log_terms = np.logaddexp(0, -z)
        # log_terms = np.log1p(np.exp(-z))
        obj = self.C * np.sum(log_terms) + np.sum(u)
        self.logloss_history.append(log_terms)
        self.loss_history.append(obj)

        return obj

    def fit(self, X, y):
        """
        Fit the L1-regularized logistic regression model.

        Args:
            X (array-like of shape (n_samples, n_features)): Training data.
            y (array-like of shape (n_samples,)): Target values.

        Returns:
            self: The fitted model.
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n_features = X.shape[1]
        u0 = np.zeros(2 * n_features)
        bounds = [(0, None)] * (2 * n_features)

        res = minimize(
            fun=lambda u: self._objective_and_grad(u, X, y),
            x0=u0,
            method=self.method,
            jac=True,
            bounds=bounds,
        )

        # if not res.success:
        #    raise RuntimeError(f"Optimization process failed: {res.message}")

        u_opt = res.x
        x_plus_opt = u_opt[:n_features]
        x_minus_opt = u_opt[n_features:]
        self.coef_ = x_plus_opt - x_minus_opt
        return self

    def predict_proba(self, X):
        """
        Predict probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        T : array-like of shape (n_samples, 2)
            Returns the probability of the sample for each class in the model,
            The columns correspond to the classes in sorted order, as they
            appear in the attribute `classes_`.
        """
        X = np.asarray(X, dtype=float)
        scores = X.dot(self.coef_)
        probs = 1 / (1 + np.exp(-scores))
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Predicted class label per sample.
        """
        X = np.asarray(X, dtype=float)
        return np.where(X.dot(self.coef_) >= 0, 1, -1)
