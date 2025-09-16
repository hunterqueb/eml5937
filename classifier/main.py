import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main(interval_ms=200):
    x, y_true, DB = createDataset(200)
    w_history = trainPerceptron(x, y_true, lr=0.001, epochs=50)

    # Run animation
    animate_training(x, y_true, DB, w_history, interval=interval_ms)
    # After animation, plot ground-truth labels for reference
    plot_final_truth(x, y_true, DB, w_history[-1])

    plt.show()

def createDataset(numDataPoints):
    scalar = 10
    # generate one extra point to define the separating ratio DB
    x = np.random.uniform(low=0.5, high=scalar, size=(numDataPoints + 1, 2))
    DB = x[-1]
    x = x[:-1]
    # label by ratio threshold: sqrt(x0/x1) > sqrt(DB0/DB1)  =>  1 else -1
    thr = np.sqrt(DB[0] / DB[1])
    y = np.where(np.sqrt(x[:, 0] / x[:, 1]) > thr, 1, -1).astype(float)
    return x, y, DB

def trainPerceptron(x, y, lr, epochs):
    w = np.array((0,0))
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

def _colors_from_labels(y):
    # 1 -> C1, -1 -> C2
    c = np.empty((len(y),), dtype=object)
    c[y >= 0] = "C1"
    c[y < 0] = "C2"
    return c

def animate_training(x, y_true, DB, w_history, interval=200):
    fig, ax = plt.subplots(figsize=(6, 6))

    # initial prediction (epoch 0)
    y_pred0 = evalPerceptron(x, *w_history[0])
    sc = ax.scatter(x[:, 0], x[:, 1], s=20, c=_colors_from_labels(y_pred0), edgecolors="none")
    # reference point defining the threshold
    ax.scatter([DB[0]], [DB[1]], c=["C0"], s=60, marker="X", zorder=3, label="DB")
    ax.set_xlabel("spring constant")
    ax.set_ylabel("mass")
    ax.set_title("Training animation — Perceptron")
    ax.legend(loc="upper right")

    # decision boundary line artist
    (line,) = ax.plot([], [], "k--", linewidth=1)

    # set reasonable limits
    pad_x = 0.1 * (x[:, 0].max() - x[:, 0].min())
    pad_y = 0.1 * (x[:, 1].max() - x[:, 1].min())
    ax.set_xlim(x[:, 0].min() - pad_x, x[:, 0].max() + pad_x)
    ax.set_ylim(x[:, 1].min() - pad_y, x[:, 1].max() + pad_y)

    def update(frame):
        w, b = w_history[frame]
        # update point colors by current predictions
        y_pred = evalPerceptron(x, w, b)
        sc.set_color(_colors_from_labels(y_pred))

        # update decision boundary: w0*x + w1*y + b = 0  =>  y = (-b - w0*x)/w1
        if np.linalg.norm(w) < 1e-12:
            line.set_data([], [])
        else:
            x_vals = np.array(ax.get_xlim())
            if abs(w[1]) > 1e-12:
                y_vals = (-b - w[0] * x_vals) / w[1]
            else:
                # nearly vertical boundary: x = -b/w0; draw a vertical line
                x0 = -b / (w[0] + 1e-12)
                x_vals = np.array([x0, x0])
                y_vals = np.array(ax.get_ylim())
            line.set_data(x_vals, y_vals)

        ax.set_title(f"Perceptron training — epoch {frame}/{len(w_history) - 1}")
        return sc, line

    ani = FuncAnimation(
        fig,
        update,
        frames=len(w_history),
        interval=interval,
        blit=False,
        repeat=False,
    )

    ani.save("classifier/perceptron.gif", writer="pillow", fps=max(1, int(1000 / interval)))

    return ani

def plot_final_truth(x, y_true, DB, final_params):
    w, b = final_params
    fig, ax = plt.subplots(figsize=(6,6))

    # true labels as given by dataset
    colors = np.where(y_true > 0, "C1", "C2")
    ax.scatter(x[:,0], x[:,1], c=colors, s=20, edgecolors="none", label="True labels")

    ax.scatter([DB[0]], [DB[1]], c=["C0"], s=60, marker="X", zorder=3, label="DB")

    # decision boundary from final weights
    if abs(w[1]) > 1e-12:
        x_vals = np.array(ax.get_xlim())
        y_vals = (-b - w[0]*x_vals) / w[1]
        ax.plot(x_vals, y_vals, "k--", linewidth=1, label="Final boundary")
    else:
        x0 = -b / (w[0] + 1e-12)
        ax.axvline(x0, color="k", linestyle="--", linewidth=1, label="Final boundary")

    ax.set_xlabel("spring constant")
    ax.set_ylabel("mass")
    ax.set_title("Final predictions vs. true labels")
    ax.legend(loc="upper right")
    fig.savefig("classifier/final_pred.png")
    return fig, ax

if __name__ == "__main__":
    main(interval_ms=150)
