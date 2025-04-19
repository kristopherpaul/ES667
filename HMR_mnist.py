import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# PCA whitening to 30 dims
pca = PCA(n_components=30, whiten=True, random_state=0)
Xw_train = pca.fit_transform(X_train)
Xw_test  = pca.transform(X_test)

# Hermite feature generator (up to D=4)
def hermite_features(Z, D):
    feats = []
    for m in range(1, D+1):
        if m == 1:
            H = Z
        elif m == 2:
            H = Z**2 - 1
        elif m == 3:
            H = Z**3 - 3*Z
        elif m == 4:
            H = Z**4 - 6*Z**2 + 3
        else:
            raise ValueError("D>4 not supported")
        H = H / np.sqrt(math.factorial(m))
        H = (H - H.mean(axis=0)) / (H.std(axis=0) + 1e-6)
        feats.append(H)
    return np.concatenate(feats, axis=1)

# Stage1: learn subspace via Hermite cross-covariance (order M)
def learn_subspace(Xw, y, k1, M):
    H = hermite_features(Xw, M)
    Cov = (H.T @ Xw) / Xw.shape[0]
    _, _, Vt = np.linalg.svd(Cov, full_matrices=False)
    return Vt[:k1].T  # (30, k1)

# Stage2: Hermite + multinomial logistic
def learn_moment_logistic(Xw, y, V1, D, C):
    Z = Xw @ V1
    H = hermite_features(Z, D)
    clf = LogisticRegression(
        multi_class='multinomial', solver='lbfgs',
        penalty='l2', C=C, max_iter=200, random_state=0
    )
    clf.fit(H, y)
    return clf

# Grid search
results = []

# baseline PCA-Logistic
for C in tqdm([0.01, 0.1, 1, 10]):
    clf = LogisticRegression(
        multi_class='multinomial', solver='lbfgs',
        penalty='l2', C=C, max_iter=200, random_state=0
    )
    clf.fit(Xw_train, y_train)
    acc = accuracy_score(y_test, clf.predict(Xw_test))
    results.append(("PCA‑Logistic", f"C={C}", acc))

# Hermite Moment Regression
for k1 in tqdm([10,20,30]):
  for D in [2,3]:
    for C in [0.01,0.1,1]:
      V1 = learn_subspace(Xw_train, y_train, k1, M=D)
      clf2 = learn_moment_logistic(Xw_train, y_train, V1, D, C)
      H_test = hermite_features(Xw_test @ V1, D)
      acc = accuracy_score(y_test, clf2.predict(H_test))
      results.append(("2L HighMoment", f"k1={k1},D={D},C={C}", acc))

df = pd.DataFrame(results, columns=["Method","Params","TestAcc"])
best = df.groupby("Method").apply(lambda g: g.nlargest(1,"TestAcc")).reset_index(drop=True)
print(best)