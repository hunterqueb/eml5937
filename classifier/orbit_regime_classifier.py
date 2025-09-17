import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

R = 6378.0

def main():
    train_dataset, train_labels, test_dataset, test_labels = load_dataset()

    W, b, history = trainPerceptronMulticlass(train_dataset, train_labels, lr=1e-3, epochs=5)

    overall, per_class = per_class_accuracy(test_dataset, test_labels, W, b)
    print("Overall accuracy:", overall)
    print("Per-class breakdown:", per_class)

    plot_per_class_accuracy(test_dataset, test_labels, history)
    compute_confusion_matrix(test_dataset, test_labels, W, b)

    for i in range(len(history)):
        plot_decision_boundaries(test_dataset, test_labels, history[i][0], history[i][1], class_labels=["leo", "heo", "meo", "geo"],epoch = i)

    plt.show()

def load_dataset():
    OENoThrust = np.load("classifier/OEArrayNoThrust.npz")["OEArrayNoThrust"]
    dataset_labels = np.load("classifier/dataset_orbit_labels.npz")["dataset_orbit_labels"]
    train_dataset, train_labels, test_dataset, test_labels = process_dataset(OENoThrust, dataset_labels)
    return train_dataset, train_labels, test_dataset, test_labels


def process_dataset(OENoThrust, dataset_labels, train_ratio=0.7):
    """
    cleans time series into just semimajor axis, processes the labels into ints, shuffles dataset, and returns a train and test set
    """
    map_lbl = {"leo": 0, "heo": 1, "meo": 2, "geo": 3}
    y = np.array([map_lbl[str(lbl).lower()] for lbl in dataset_labels], dtype=np.int64)

    N = y.shape[0]
    SMA = (OENoThrust[:, 0, 0] / R).reshape((N, 1))
    ECC = (OENoThrust[:, 0, 1] * 10).reshape((N, 1))

    feat = np.hstack((SMA, ECC))

    idx = np.random.permutation(N)
    n_train = int(np.floor(train_ratio * N))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X_train = feat[train_idx]
    y_train = y[train_idx]
    X_test = feat[test_idx]
    y_test = y[test_idx]
    return X_train, y_train, X_test, y_test


def trainPerceptronMulticlass(X, y, lr=1e-3, epochs=20, num_classes=None):
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    N, D = X.shape
    K = int(np.max(y)) + 1 if num_classes is None else int(num_classes)

    W = np.zeros((K, D), dtype=float)
    b = np.zeros(K, dtype=float)

    history = [(W.copy(), b.copy())]  # epoch 0

    for epoch in range(epochs):
        for i in range(N):
            xi = X[i]
            yi = y[i]
            scores = W @ xi + b  # (K,)
            y_pred = int(np.argmax(scores))
            if y_pred != yi:
                # reward
                W[yi] += lr * xi
                b[yi] += lr
                # penalty
                W[y_pred] -= lr * xi
                b[y_pred] -= lr
        history.append((W.copy(), b.copy()))

    return W, b, history


def predict_multiclass(X, W, b):
    X = np.asarray(X)
    scores = X @ W.T + b
    return np.argmax(scores, axis=1)

def per_class_accuracy(X, y, W, b):
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    preds = np.argmax(X @ W.T + b, axis=1)

    overall_acc = np.mean(preds == y)
    per_class_acc = {}
    K = int(np.max(y)) + 1
    for c in range(K):
        mask = (y == c)
        per_class_acc[c] = np.mean(preds[mask] == y[mask]) if np.any(mask) else np.nan
    return overall_acc, per_class_acc


def plot_per_class_accuracy(X, y, history):
    K = int(np.max(y)) + 1
    epochs = len(history)
    acc_per_class = np.zeros((epochs, K))

    for e, (W, b) in enumerate(history):
        preds = np.argmax(X @ W.T + b, axis=1)
        for c in range(K):
            mask = (y == c)
            acc_per_class[e, c] = np.mean(preds[mask] == y[mask]) if np.any(mask) else np.nan

    class_labels = ["leo", "heo", "meo", "geo"]
    plt.figure()
    for c in range(K):
        plt.plot(range(epochs), acc_per_class[:, c], label=class_labels[c])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def compute_confusion_matrix(X, y, W, b, normalize=True, plot=True):
    preds = np.argmax(X @ W.T + b, axis=1)

    cm = confusion_matrix(y, preds, labels=np.arange(np.max(y)+1), normalize='true' if normalize else None)

    if plot:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['leo', 'heo', 'meo', 'geo'])
        disp.plot(cmap=plt.cm.Blues, values_format=".2f" if normalize else "d")
        plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    return cm


def plot_decision_boundaries(X, y, W, b, class_labels=None,epoch = None):
    K, D = W.shape
    padding=0.08
    resolution=400
    scatter_alpha=0.7

    # Identify SMA/ECC columns
    if D == 3:
        sma = X[:, 1]
        ecc = X[:, 2]
    else:
        sma = X[:, 0]
        ecc = X[:, 1]

    # Bounds with padding
    def _pad(lo, hi, frac):
        span = max(hi - lo, 1e-6)
        return lo - frac * span, hi + frac * span

    sma_min, sma_max = _pad(float(np.min(sma)), float(np.max(sma)), padding)
    ecc_min, ecc_max = _pad(float(np.min(ecc)), float(np.max(ecc)), padding)

    xs = np.linspace(sma_min, sma_max, resolution)
    ys = np.linspace(ecc_min, ecc_max, resolution)
    XX, YY = np.meshgrid(xs, ys)

    if D == 3:
        grid = np.column_stack([np.ones(XX.size), XX.ravel(), YY.ravel()])
    else:
        grid = np.column_stack([XX.ravel(), YY.ravel()])

    Z = np.argmax(grid @ W.T + b, axis=1).reshape(XX.shape)

    if class_labels is None:
        class_labels = [str(i) for i in range(K)]

    # Discrete colormap with up to 8 distinct colors (extend if you add classes)
    cmap = ListedColormap(["C0", "C1", "C2", "C3",])

    plt.figure()

    # Filled regions
    plt.contourf(XX, YY, Z, levels=np.arange(K + 1) - 0.5, alpha=0.3, cmap=cmap, antialiased=True)

    # Boundary lines
    plt.contour(XX, YY, Z, levels=np.arange(K) + 0.5, colors="k", linewidths=0.6, alpha=0.7)

    # Scatter points
    for c in range(K):
        mask = (y == c)
        plt.scatter(sma[mask], ecc[mask], s=12, edgecolors="k", linewidths=0.2,alpha=scatter_alpha, label=class_labels[c])

    plt.xlabel("SMA / R")
    plt.ylabel("Eccentricity x 10")
    if epoch is None:
        plt.title("Decision Regions")
    else:
        plt.title(f"Decision Regions at Epoch {epoch}")
    plt.legend(markerscale=1.5, fontsize=9, frameon=True)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()


if __name__ == "__main__":
    main()
