import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms

# Load MNIST
def load_mnist_numpy():
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root=".", train=False, download=True, transform=transform)
    X_train = train_ds.data.view(-1, 28*28).float().numpy()
    y_train = train_ds.targets.numpy()
    X_test  = test_ds.data.view(-1, 28*28).float().numpy()
    y_test  = test_ds.targets.numpy()
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_mnist_numpy()

# Moment‑based one‑layer classifier
def learn_one_layer_clf(X, y, k):
    N = X.shape[0]
    M2 = (X * y[:, None]).T @ X / N            # (d×d) second moment
    eigvals, eigvecs = np.linalg.eigh(M2)
    idx = np.argsort(-np.abs(eigvals))[:k]
    V = eigvecs[:, idx]                       # d×k
    H = np.maximum(X @ V, 0)                  # ReLU features
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500
    ).fit(H, y)
    return V, clf

# Hierarchical Moment Decomposition
def learn_two_layer_clf(X, y, k1, k2):
    V1, clf1 = learn_one_layer_clf(X, y, k1)
    H1 = np.maximum(X @ V1, 0)
    V2, clf2 = learn_one_layer_clf(H1, y, k2)
    return (V1, clf1), (V2, clf2)

# Run experiments
d, k1, k2 = 28*28, 50, 20

# 1‑Layer Moment
V1, clf1 = learn_one_layer_clf(X_train, y_train, k1)
train_acc1 = accuracy_score(
    y_train,
    clf1.predict(np.maximum(X_train @ V1, 0))
)
test_acc1  = accuracy_score(
    y_test,
    clf1.predict(np.maximum(X_test  @ V1, 0))
)

# 2‑Layer Moment
(params1, params2) = learn_two_layer_clf(X_train, y_train, k1, k2)
V2, clf2 = params2
train_acc2 = accuracy_score(
    y_train,
    clf2.predict(
        np.maximum(
            np.maximum(X_train @ params1[0], 0) @ V2,
        0)
    )
)
test_acc2 = accuracy_score(
    y_test,
    clf2.predict(
        np.maximum(
            np.maximum(X_test @ params1[0], 0) @ V2,
        0)
    )
)

import pandas as pd
results = pd.DataFrame({
    'Method': [
        '1‑Layer Moment',
        'HMD'
    ],
    'Train Acc.': [train_acc1, train_acc2],
    'Test Acc.':  [test_acc1,  test_acc2]
})

print(results)