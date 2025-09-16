import numpy as np
import matplotlib.pyplot as plt


def main():

    train_dataset,train_dataset_labels,test_dataset,test_dataset_labels = load_dataset()

    plt.figure()
    plt.plot(train_dataset)
    

    
    
    plt.show()
    return


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
    SMA = OENoThrust[:,0,0]

    map_lbl = {"leo": 0, "meo": 1, "geo": 2}
    y = np.array([map_lbl[str(lbl).lower()] for lbl in dataset_labels], dtype=np.int64)

    N = y.shape[0]
    if SMA.shape[0] != N:
        raise ValueError(f"SMA and labels mismatch: {SMA.shape[0]} vs {N}")

    # 3) simple permutation split (no grouping)
    idx = np.random.permutation(N)
    n_train = int(np.floor(train_ratio * N))
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]

    X_train = SMA[train_idx]/R
    y_train = y[train_idx]
    X_test  = SMA[test_idx]/R
    y_test  = y[test_idx]

    return X_train, y_train, X_test, y_test

def trainPerceptron(x, y, lr, epochs):
    w = np.array((0))
    b = 1
    w_history = [(w.copy(), float(b))]  # epoch 0 state

    for epoch in range(epochs):
        for i in range(len(y)):
            prediction = np.sign(np.dot(w,x[i]) + b)
            if prediction != y[i]:
                w = w + lr * y[i] * x[i]
                b = b + lr * y[i]
        w_history.append((w.copy(), float(b)))
    return w_history

def evalPerceptron(x, w, b):
    return np.sign(x @ w + b)



if __name__ == "__main__":
    main()
    pass