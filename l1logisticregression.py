import numpy as np
from scipy.optimize import minimize
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
    
class L1LogisticRegression:
    def __init__(self, C: int = 100, method: str = "L-BFGS-B"):
        self.C = C
        self.method = method
        self.coef_ = None

    def _reparametrization(self, X):
        X = np.asarray(X)
        X_plus = np.maximum(X, 0)
        X_minus = np.maximum(-X, 0)

        return X_plus, X_minus

    def _objective(self, u, X, y):
        X_plus, _ = self._reparametrization(X)
        n = X_plus.shape[1]
        x_plus = u[:n]
        x_minus = u[n:]
        x = x_plus - x_minus
        z = y * (X.dot(x))
        log_terms = np.log1p(np.exp(-z))

        return self.C * np.sum(log_terms) + np.sum(u)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n_features = X.shape[1]
        u0 = np.zeros(2 * n_features)
        bounds = [(0, None)] * (2 * n_features) 

        def func(u):
            return self._objective(u, X, y)

        res = minimize(func, u0, bounds=bounds, method=self.method)

        if not res.success:
            raise RuntimeError(f"Optimization process failed: {res.message}")

        u_opt = res.x
        x_plus_opt = u_opt[:n_features]
        x_minus_opt = u_opt[n_features:]
        self.coef_ = x_plus_opt - x_minus_opt
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        scores = X.dot(self.coef_)
        probs = 1 / (1 + np.exp(-scores))
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.where(X.dot(self.coef_) >= 0, 1, -1)
    

# Demo 
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Generate synthetic data
    X, y = make_classification(
        n_samples=500, n_features=20,
        n_informative=5, n_redundant=2,
        random_state=42
    )
    # Convert labels to +1/-1
    y = np.where(y == 1, 1, -1)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Train model
    model = L1LogisticRegression(C=100)
    model.fit(X_train, y_train)

    # Predict and evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    nonzero_coefs = np.sum(model.coef_ != 0)

    # Print results
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Non-zero coefficients: {nonzero_coefs}")
    print(f"Coefficients: {model.coef_}")


