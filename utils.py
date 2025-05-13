from typing import Any, Dict, List, Sequence, Tuple
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from l1logisticregression import L1LogisticRegression
from dataset_utils import load_ucirepo_dataset


def run_and_collect_histories(
    X_train, y_train, X_test, y_test, C: float, methods: list[str]
) -> dict[str, list[float]]:
    """
    Train L1LogisticRegression for each method in `methods`, collect its loss history.

    Returns:
        histories: dict mapping method name → model.loss_history list
    """
    histories: dict[str, list[float]] = {}
    for method in methods:
        print(f"Training with method = {method!r}")
        model = L1LogisticRegression(C=C, method=method)
        model.fit(X_train, y_train)

        # evaluate (optional)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

        # collect the per-iteration loss history
        histories[method] = np.log(model.loss_history.copy())
    return histories


def plot_objective_histories(
    loss_histories: dict[str, list[float]],
    title: str = "Log Objective History",
    xlabel: str = "Iteration",
    ylabel: str = "Log Objective",
    xlim: int = 10000,
) -> None:
    plt.figure()
    for name, hist in loss_histories.items():
        plt.plot(hist, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, xlim)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_c_experiment_logloss(
    Cs: List[float],
    method: str = "L-BFGS-B",
    n_repeats: int = 10,
    test_size: float = 0.2,
    random_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each C in Cs, repeat training/testing n_repeats times, compute mean & std log-loss.

    Returns:
        Cs_arr:      np.ndarray of C values
        mean_losses: np.ndarray of mean log-loss per C
        std_losses:  np.ndarray of std deviation of log-loss per C
    """
    # Generate synthetic classification data once
    X, y = load_ucirepo_dataset(17)
    # remap labels {0,1} -> {-1,1}
    y = np.where(y == 1, 1, -1)

    mean_losses = []
    std_losses = []

    for C in Cs:
        losses = []
        for i in range(n_repeats):
            # split with different random_state each repeat
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=test_size, random_state=random_seed + i
            )

            model = L1LogisticRegression(C=C, method=method)
            model.fit(X_train, y_train)

            losses.append(model.logloss_history[-1])

        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))
        print(f"C={C:.4g} → mean log-loss={mean_losses[-1]:.4f} ±{std_losses[-1]:.4f}")

    Cs_arr = np.array(Cs)
    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)

    best_idx = np.argmin(mean_losses)
    print(
        f"\nBest C = {Cs_arr[best_idx]:.4g} with mean log-loss = {mean_losses[best_idx]:.4f}"
    )

    return Cs_arr, mean_losses, std_losses


def plot_mean_log_loss_vs_C(
    Cs: Sequence[float],
    mean_losses: Sequence[float],
    std_losses: Sequence[float],
    x_label: str = "Regularization parameter C",
    y_label: str = "Mean Log-Loss",
    title: str = "Mean Log-Loss vs. C",
    capsize: int = 3,
) -> None:
    """
    Plot mean log-loss with error bars as a function of C and display the plot.

    Args:
        Cs: Sequence of C values.
        mean_losses: Corresponding mean log-loss values.
        std_losses: Corresponding standard deviations of log-loss.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        title: Plot title.
        capsize: Error bar cap size.
    """
    plt.figure()
    plt.errorbar(Cs, mean_losses, yerr=std_losses, marker="o", capsize=capsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()


def run_c_experiment_accuracy(
    Cs: List[float],
    method: str = "L-BFGS-B",
    n_repeats: int = 10,
    test_size: float = 0.2,
    random_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each C in Cs, repeat training/testing n_repeats times, compute mean & std test accuracy.

    Returns:
        Cs_arr:       np.ndarray of C values
        mean_accs:    np.ndarray of mean accuracy per C
        std_accs:     np.ndarray of std deviation of accuracy per C
    """
    # Load dataset once
    X, y = load_ucirepo_dataset(17)  # load features X and labels y in {-1,+1}

    mean_accs = []
    std_accs = []

    for C in Cs:
        accs = []
        for i in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed + i
            )

            model = L1LogisticRegression(C=C, method=method)
            model.fit(X_train, y_train)

            # Predict labels and compute accuracy
            y_pred = model.predict(X_test)
            acc = np.mean(y_pred == y_test)
            accs.append(acc)

        mean_accs.append(np.mean(accs))
        std_accs.append(np.std(accs))
        print(f"C={C:.4g} → mean accuracy={mean_accs[-1]:.4f} ±{std_accs[-1]:.4f}")

    Cs_arr = np.array(Cs)
    mean_accs = np.array(mean_accs)
    std_accs = np.array(std_accs)

    best_idx = np.argmax(mean_accs)
    print(
        f"\nBest C = {Cs_arr[best_idx]:.4g} with mean accuracy = {mean_accs[best_idx]:.4f}"
    )

    return Cs_arr, mean_accs, std_accs


def plot_mean_accuracy_vs_C(
    Cs: Sequence[float],
    mean_accs: Sequence[float],
    std_accs: Sequence[float],
    x_label: str = "Regularization parameter C",
    y_label: str = "Mean Accuracy",
    title: str = "Mean Accuracy vs. C",
    capsize: int = 3,
) -> None:
    """
    Plot mean test accuracy with error bars as a function of C and display the plot.

    Args:
        Cs: Sequence of C values.
        mean_accs: Corresponding mean accuracy values.
        std_accs: Corresponding standard deviations of accuracy.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        title: Plot title.
        capsize: Error bar cap size.
    """
    plt.figure()
    plt.errorbar(Cs, mean_accs, yerr=std_accs, marker="o", capsize=capsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
