import numpy as np
import matplotlib.pyplot as plt
def main():
    
    x,y,DB = createDataset(200)

    w,b = trainPerceptron(x,y,0.001,20,DB)

    plotFeatureSpace(x,y,DB)
    plt.title("Perfect Prediction")

    plt.show()
    return

def createDataset(numDataPoints):
    scalar = 10
    x = np.zeros((numDataPoints,2))
    y = np.zeros((numDataPoints,))

    x =  np.random.uniform(low=0.5, high=scalar, size=(numDataPoints+1,2))
    DB = x[-1]
    x = x[:-1]
    for i in range(numDataPoints):
        y[i] = 1 if np.sqrt(x[i][0]/x[i][1]) > np.sqrt(DB[0]/DB[1]) else -1

    return x,y,DB

def plotFeatureSpace(x,y,DB):
    plt.figure()
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(x[i][0],x[i][1],color="C1")
        else:
            plt.scatter(x[i][0],x[i][1],color="C2")
    plt.scatter(DB[0],DB[1],color="C0")
    plt.xlabel("spring constant")
    plt.ylabel("mass")

def trainPerceptron(x,y,lr,epochs,DB):
    w = np.array([0,0])
    b = 1
    for epoch in range(epochs):
        for i in range(len(y)):
            prediction = np.sign(np.dot(w,x[i]) + b)
            if prediction != y[i]:
                w = w + lr * y[i] * x[i]
                b = b + lr * y[i]
        y_eval = evalPerceptron(x,w,b)
        plotFeatureSpace(x,y_eval,DB)

    return w, b

def evalPerceptron(x,w,b):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = np.sign(np.dot(w,x[i]) + b)
    return y

if __name__ == "__main__":
    main()