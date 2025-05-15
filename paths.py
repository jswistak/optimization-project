import numpy as np
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64
from sklearn.linear_model import LogisticRegression
from typing import NamedTuple

from l1logisticregression import (
    L1LogisticRegression,
)

rng = Generator(PCG64(seed=0))


class Dataset(NamedTuple):
    observations: np.ndarray
    labels: np.ndarray

    @classmethod
    def get_simulation_data(cls, n: int, feats: int) -> "Dataset":
        observations = rng.normal(0, 1, size=(n, feats + 3))

        observations[:, 0] = rng.normal(3, 3, size=n)
        observations[:, 4] = rng.normal(1, 5, size=n)
        observations[:, 5] = rng.normal(-1, 7, size=n)
        observations[:, 6] = rng.normal(-1, 7, size=n)
        observations[:, 7] = rng.normal(-1, 7, size=n)
        # true intercept and linear term on first 4 cols
        beta_0 = 0.5
        logits = beta_0 + observations[:, :4].sum(axis=1)
        p = 1 / (1 + np.exp(-logits))
        labels = rng.binomial(1, p, size=n)
        return cls(observations, labels)


if __name__ == "__main__":
    n_obs = 500
    n_feats = 5
    max_iter = 1000
    lambda_min, lambda_max, n_lambda = 1e-2, 1e3, 100
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambda)

    train = Dataset.get_simulation_data(n_obs, n_feats)
    val = Dataset.get_simulation_data(100, n_feats)

    coefs_sklearn = []
    coefs_custom = []

    skl = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        fit_intercept=False,
        max_iter=max_iter,
        tol=1e-5,
        random_state=0,
    )

    for lam in lambdas:
        C = 1.0 / lam

        skl.set_params(C=C)
        skl.fit(train.observations, train.labels)
        coefs_sklearn.append(skl.coef_.ravel().copy())

        clf = L1LogisticRegression(C=C, method="L-BFGS-B")
        clf.fit(train.observations, train.labels)
        coefs_custom.append(clf.coef_.copy())

    coefs_sklearn = np.array(coefs_sklearn)
    coefs_custom = np.array(coefs_custom)

    # plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    for i in range(coefs_custom.shape[1]):
        axes[0].plot(lambdas, coefs_custom[:, i], marker=".", label=f"Feat {i + 1}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("λ")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title("Custom L1LogisticRegression")
    axes[0].legend(fontsize=12)

    for i in range(coefs_sklearn.shape[1]):
        axes[1].plot(lambdas, coefs_sklearn[:, i], marker=".", label=f"Feat {i + 1}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("λ")
    axes[1].set_title("sklearn LogisticRegression")
    axes[1].legend(fontsize=12)

    plt.suptitle("Coefficient Paths vs Regularization Strength", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
