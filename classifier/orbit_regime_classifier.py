import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():

    train_dataset,train_dataset_labels,test_dataset,test_dataset_labels = load_dataset()
    
    classifier = multiclassPerceptron(num_classes=4,num_features=train_dataset.shape[1])

    classifier.train(train_dataset,train_dataset_labels,lr = 1,epochs=10)

    overall, per_class = classifier.per_class_accuracy(test_dataset, test_dataset_labels)
    print("Overall accuracy:", overall)
    print("Per-class breakdown:", per_class)
    
    classifier.plot_per_class_accuracy(test_dataset,test_dataset_labels)
    classifier.compute_confusion_matrix(test_dataset,test_dataset_labels)

    print(classifier.history)

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

    SMA = SMA / R
    x0 = np.ones((N))
    SMA = np.column_stack((x0,SMA))

    idx = np.random.permutation(N)
    n_train = int(np.floor(train_ratio * N))
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]

    X_train = SMA[train_idx]
    y_train = y[train_idx]
    X_test  = SMA[test_idx]
    y_test  = y[test_idx]
    return X_train, y_train, X_test, y_test


class multiclassPerceptron():
    def __init__(self,num_classes,num_features):
        self.K = num_classes
        self.D = num_features

    def forward(self,X,perClass=False):
        # X = np.asarray(X)
        scores = X @ self.W.T + self.b        # (N, K)
        return np.argmax(scores,axis=1) if perClass else int(np.argmax(scores)) 
    
    def train(self,X, y, lr=1e-3, epochs=20):
        """
        Multiclass perceptron (one-vs-rest style updates).

        X: (N, D) features
        y: (N,) integer class labels in [0, K-1]
        lr: learning rate
        epochs: passes over data
        num_classes: if None, inferred from y
        Returns: W (K, D), b (K,), history (list of (W, b) per epoch)
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.int64)
        N, _ = X.shape

        # params
        self.W = np.zeros((self.K, self.D), dtype=float)
        self.b = np.zeros(self.K, dtype=float)

        self.history = [(self.W.copy(), self.b.copy())]  # epoch 0

        for epoch in range(epochs):
            idx = np.arange(N)

            for i in idx:
                xi = X[i]
                yi = y[i]
                y_pred = self.forward(xi)
                if y_pred != yi:
                    # promote true class, demote predicted class
                    self.W[yi] += lr * xi
                    self.b[yi] += lr
                    self.W[y_pred] -= lr * xi
                    self.b[y_pred] -= lr

            self.history.append((self.W.copy(), self.b.copy()))   

    def per_class_accuracy(self,X, y):
        """
        Compute per-class and overall accuracy.

        X: (N, D) features
        y: (N,) integer labels
        W: (K, D) weight matrix
        b: (K,) bias vector

        Returns:
            overall_acc (float),
            per_class_acc (dict: class_id -> accuracy)
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.int64)
        preds = self.forward(X,perClass=True)

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


    def plot_per_class_accuracy(self,X, y):
        """
        Plot per-class accuracy across epochs.

        X: (N, D) features
        y: (N,) true integer labels
        history: list of (W, b) tuples per epoch, as returned by trainPerceptron_multiclass
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.int64)
        K = int(np.max(y)) + 1
        epochs = len(self.history)

        # store accuracies: shape (epochs, K)
        acc_per_class = np.zeros((epochs, K))

        for e, (W, b) in enumerate(self.history):
            preds = self.forward(X,perClass=True)
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

    def compute_confusion_matrix(self,X, y, normalize=True, plot=True):
        """
        Compute and optionally plot the confusion matrix.

        X: (N, D) features
        y: (N,) true integer labels
        W: (K, D) weight matrix
        b: (K,) bias vector
        normalize: if True, rows are normalized to sum to 1
        plot: if True, display the matrix with matplotlib

        Returns:
            cm (K, K) numpy array (confusion matrix)
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.int64)
        preds = self.forward(X,perClass=True)

        cm = confusion_matrix(y, preds, labels=np.arange(np.max(y)+1), normalize='true' if normalize else None)

        if plot:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(np.max(y)+1))
            disp.plot(cmap=plt.cm.Blues, values_format=".2f" if normalize else "d")
            plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

        return cm



if __name__ == "__main__":
    main()
