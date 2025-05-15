import time
from typing import Any, Dict, List, Sequence, Tuple
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from l1logisticregression import L1LogisticRegression
from dataset_utils import generate_high_dim_classification, load_ucirepo_dataset


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
    """
    Plots the objective (loss) histories for multiple optimization runs.

    Args:
        loss_histories (dict[str, list[float]]): A dictionary where keys are the names of the optimization runs 
            and values are lists of loss values recorded at each iteration.
        title (str, optional): The title of the plot. Defaults to "Log Objective History".
        xlabel (str, optional): The label for the x-axis. Defaults to "Iteration".
        ylabel (str, optional): The label for the y-axis. Defaults to "Log Objective".
        xlim (int, optional): The maximum limit for the x-axis. Defaults to 10000.

    Returns:
        None: This function does not return anything. It displays the plot.

    Notes:
        - The function uses matplotlib to create the plot.
        - Each optimization run is plotted as a separate line with a legend entry.
        - The x-axis is limited to the range [0, xlim].
        - The plot includes a grid and is displayed with a tight layout.
    """
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


def run_features_number_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float,
    methods: Sequence[str],
    features_numbers: Sequence[int],
    n_repeats: int = 3,
    random_state: int = 0
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Runs L1-logistic experiments across different feature-subset sizes,
    repeating each fit `n_repeats` times to compute mean & std of:
      - log_loss
      - fit_time

    Returns a dict:
        { features_number: {
            method_name: {
              'log_loss_mean': float,
              'log_loss_std' : float,
              'fit_time_mean': float,
              'fit_time_std' : float
            }
        }}
    """
    rng = np.random.RandomState(random_state)
    n_total_features = X_train.shape[1]
    histories: Dict[int, Dict[str, Dict[str, float]]] = {}

    for n_feat in features_numbers:
        histories[n_feat] = {}

        for method in methods:
            log_losses = []
            times = []

            for rep in range(n_repeats):
                idx = rng.choice(n_total_features, size=n_feat, replace=False)
                X_sub = X_train[:, idx]

                model = L1LogisticRegression(C=C, method=method)
                
                t0 = time.time()
                model.fit(X_sub, y_train)
                dt = time.time() - t0

                log_losses.append(np.log(model.loss_history[-1]))
                times.append(dt)

            # compute stats
            ll_mean = float(np.mean(log_losses))
            ll_std  = float(np.std(log_losses, ddof=1))
            t_mean  = float(np.mean(times))
            t_std   = float(np.std(times, ddof=1))

            histories[n_feat][method] = {
                'log_loss_mean': ll_mean,
                'log_loss_std' : ll_std,
                'fit_time_mean': t_mean,
                'fit_time_std' : t_std
            }
            print(f"n_feat={n_feat}, method={method}  "
                  f"log_loss={ll_mean:.4f}±{ll_std:.4f}  "
                  f"fit_time={t_mean:.3f}s±{t_std:.3f}s")

    return histories

def plot_metric_vs_features(
    features_numbers: Sequence[int],
    histories: Dict[int, Dict[str, Dict[str, float]]],
    metric: str = "log_loss",
    methods: Sequence[str] = None,
    x_label: str = "Number of Features",
    y_label: str = None,
    title: str = None,
    capsize: int = 3
) -> None:
    """
    Plot the mean and standard deviation of a chosen metric
    ('log_loss' or 'fit_time') as a function of number of features
    for each method.

    Args:
        features_numbers: Sequence of feature‐set sizes (ints).
        histories: Mapping from features_number to
                   {method_name: {
                       metric_mean: float,
                       metric_std : float
                   }}.
        metric:       Base metric to plot ('log_loss' or 'fit_time').
        methods:      If provided, only plot these methods; otherwise infer all.
        x_label:      Label for the x-axis.
        y_label:      Label for the y-axis (defaults to metric title).
        title:        Plot title (defaults to "<Metric> vs. Number of Features").
        capsize:      Error bar cap size.
        marker:       Marker style for the data points.
    """
    feats_sorted = sorted(features_numbers)

    # infer methods if not given
    if methods is None:
        methods = sorted({m for h in histories.values() for m in h.keys()})

    # Prepare labels
    if y_label is None:
        y_label = metric.replace("_", " ").title()
    if title is None:
        title = f"{y_label} vs. Number of Features"

    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"

    plt.figure()
    for method in methods:
        means = [histories[f][method][mean_key] for f in feats_sorted]
        stds = [histories[f][method][std_key] for f in feats_sorted]
        plt.errorbar(feats_sorted, means, yerr=stds, capsize=capsize, label=method)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend()
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


def run_c_experiment_sparsity(
    Cs: List[float],
    methods: List[str],
    n_repeats: int = 10,
    test_size: float = 0.2,
    random_seed: int = 0,
) -> Tuple[
    np.ndarray,
    Dict[str, np.ndarray], Dict[str, np.ndarray],
    Dict[str, np.ndarray], Dict[str, np.ndarray]
]:
    """
    For each C in Cs and each method in methods, repeat training n_repeats times
    on random train/test splits, compute mean & std of the fractions of zeroed
    x_plus and x_minus params separately.

    Returns:
        Cs_arr      : np.ndarray of C values
        mean_plus   : dict mapping method -> np.ndarray of mean fraction_zero_plus_
        std_plus    : dict mapping method -> np.ndarray of std  fraction_zero_plus_
        mean_minus  : dict mapping method -> np.ndarray of mean fraction_zero_minus_
        std_minus   : dict mapping method -> np.ndarray of std  fraction_zero_minus_
    """
    # generate data once
    X, y = generate_high_dim_classification(
        n_samples=100,
        n_features=200,
        n_informative=175,
        random_state=42
    )
    y = np.where(y == 1, 1, -1)

    # prepare accumulators
    plus_vals: Dict[str, List[float]]  = {m: [] for m in methods}
    minus_vals: Dict[str, List[float]] = {m: [] for m in methods}

    mean_plus:  Dict[str, List[float]] = {m: [] for m in methods}
    std_plus:   Dict[str, List[float]] = {m: [] for m in methods}
    mean_minus: Dict[str, List[float]] = {m: [] for m in methods}
    std_minus:  Dict[str, List[float]] = {m: [] for m in methods}

    for C in Cs:
        for method in methods:
            plus_vals[method].clear()
            minus_vals[method].clear()

            for rep in range(n_repeats):
                X_train, _, y_train, _ = train_test_split(
                    X, y, test_size=test_size,
                    random_state=random_seed + rep
                )
                clf = L1LogisticRegression(C=C, method=method)
                clf.fit(X_train, y_train)

                plus_vals[method].append(clf.fraction_zero_plus_)
                minus_vals[method].append(clf.fraction_zero_minus_)

            # compute stats for x_plus
            mp = float(np.mean(plus_vals[method]))
            sp = float(np.std(plus_vals[method], ddof=1))
            mean_plus[method].append(mp)
            std_plus[method].append(sp)

            # compute stats for x_minus
            mm = float(np.mean(minus_vals[method]))
            sm = float(np.std(minus_vals[method], ddof=1))
            mean_minus[method].append(mm)
            std_minus[method].append(sm)

            print(
                f"C={C:.4g}, method={method}  "
                f"x_plus_zero={mp:.4f}±{sp:.4f}  "
                f"x_minus_zero={mm:.4f}±{sm:.4f}"
            )

    # convert to arrays
    Cs_arr = np.array(Cs)
    for d in (mean_plus, std_plus, mean_minus, std_minus):
        for method in methods:
            d[method] = np.array(d[method])

    return Cs_arr, mean_plus, std_plus, mean_minus, std_minus


def plot_fraction_zero_vs_C(
    Cs: Sequence[float],
    mean_plus: Dict[str, Sequence[float]],
    std_plus: Dict[str, Sequence[float]],
    mean_minus: Dict[str, Sequence[float]],
    std_minus: Dict[str, Sequence[float]],
    methods: Sequence[str] = None,
    x_label: str = "Regularization parameter C",
    y_label: str = "Mean fraction of zero coefficients",
    title: str = "Sparsity (plus/minus) vs. C"
) -> None:
    """
    Plot mean fractions of zeroed x_plus (solid) and x_minus (dashed)
    coefficients with error bars as a function of C, for each method.
    """
    if methods is None:
        methods = list(mean_plus.keys())

    plt.figure()
    for method in methods:
        # solid line for x_plus
        plt.errorbar(
            Cs,
            mean_plus[method],
            linestyle="-",
            label=f"{method} (x_plus)"
        )
        # dashed line for x_minus
        plt.errorbar(
            Cs,
            mean_minus[method],
            linestyle="--",
            label=f"{method} (x_minus)"
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()