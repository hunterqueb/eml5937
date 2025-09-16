import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():

    train_dataset,train_dataset_labels,test_dataset,test_dataset_labels = load_dataset()
    
    W,b,history = trainPerceptronMulticlass(train_dataset,train_dataset_labels,lr = 1e-3,epochs=5)

    overall, per_class = per_class_accuracy(test_dataset, test_dataset_labels, W, b)
    print("Overall accuracy:", overall)
    print("Per-class breakdown:", per_class)
    
    plot_per_class_accuracy(test_dataset,test_dataset_labels,history)
    compute_confusion_matrix(test_dataset,test_dataset_labels, W, b)
    plt.show()

def load_dataset():
    OENoThrust = np.load("classifier/OEArrayNoThrust.npz")["OEArrayNoThrust"]
    dataset_labels = np.load("classifier/dataset_orbit_labels.npz")["dataset_orbit_labels"]
    train_dataset,train_dataset_labels,test_dataset,test_dataset_labels = process_dataset(OENoThrust,dataset_labels)
    return train_dataset,train_dataset_labels,test_dataset,test_dataset_labels

def process_dataset(OENoThrust,dataset_labels,train_ratio = 0.7):
    '''
    cleans time series into just semimajor axis, processes the labels into ints, shuffles dataset, and returns a train and test set
    '''
    R = 6378

    map_lbl = {"leo": 0, "heo": 1 , "meo": 2, "geo": 3}
    y = np.array([map_lbl[str(lbl).lower()] for lbl in dataset_labels], dtype=np.int64)
    
    N = y.shape[0]
    SMA = OENoThrust[:,0,0].reshape((N,1))
    ECC = OENoThrust[:,0,1].reshape((N,1))
    SMA = SMA / R
    ECC = ECC * 10
    x0 = np.ones((N))
    feat = np.column_stack((x0,SMA,ECC))

    idx = np.random.permutation(N)
    n_train = int(np.floor(train_ratio * N))
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]

    X_train = feat[train_idx]
    y_train = y[train_idx]
    X_test  = feat[test_idx]
    y_test  = y[test_idx]
    return X_train, y_train, X_test, y_test

def trainPerceptronMulticlass(X, y, lr=1e-3, epochs=20, num_classes=None):
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    N, D = X.shape
    K = int(np.max(y)) + 1 if num_classes is None else int(num_classes)

    # params
    W = np.zeros((K, D), dtype=float)
    b = np.zeros(K, dtype=float)

    history = [(W.copy(), b.copy())]  # epoch 0

    for epoch in range(epochs):
        idx = np.arange(N)

        for i in idx:
            xi = X[i]
            yi = y[i]
            scores = W @ xi + b               # (K,)
            y_pred = int(np.argmax(scores))
            if y_pred != yi:
                # promote true class, demote predicted class
                W[yi] += lr * xi
                b[yi] += lr
                W[y_pred] -= lr * xi
                b[y_pred] -= lr

        history.append((W.copy(), b.copy()))

    return W, b, history

def predict_multiclass(X, W, b):
    """X: (N, D). Returns predicted labels (N,)."""
    X = np.asarray(X)
    scores = X @ W.T + b        # (N, K)
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
        if np.any(mask):
            per_class_acc[c] = np.mean(preds[mask] == y[mask])
        else:
            per_class_acc[c] = np.nan  # no samples of this class

    return overall_acc, per_class_acc


def plot_per_class_accuracy(X, y, history):
    K = int(np.max(y)) + 1
    epochs = len(history)

    # store accuracies: shape (epochs, K)
    acc_per_class = np.zeros((epochs, K))

    for e, (W, b) in enumerate(history):
        preds = np.argmax(X @ W.T + b, axis=1)
        for c in range(K):
            mask = (y == c)
            if np.any(mask):
                acc_per_class[e, c] = np.mean(preds[mask] == y[mask])
            else:
                acc_per_class[e, c] = np.nan  # no samples of this class
    class_labels = ["leo","heo","meo","geo"]

    # plot
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
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['leo','heo','meo','geo'])
        disp.plot(cmap=plt.cm.Blues, values_format=".2f" if normalize else "d")
        plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
        plt.show()

    return cm



if __name__ == "__main__":
    main()
