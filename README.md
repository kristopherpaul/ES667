# Efficiently Learning Two‑Hidden‑Layer ReLU Networks

> **ES 667: Project Presentation**  
> _Mrigankashekhar Shandilya (22110157) & Kristopher Paul (22110125)_  
> 19 April 2025

---

## Table of Contents

- [Overview](#overview)  
- [Repository Structure](#repository-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Experimentation & Datasets](#experimentation--datasets)  
- [Results](#results)  
- [Future Work & Limitations](#future-work--limitations)  
- [References](#references)

---

## Overview

We study the problem of **efficiently PAC‑learning** a **two‑hidden‑layer ReLU network** under the Gaussian input model. 


<!--Concretely, given
\[
F(x) \;=\; \sum_{j=1}^{k_2} b_j \,\mathrm{ReLU}\Bigl(\sum_{i=1}^{k_1} a_{ji}\,\mathrm{ReLU}(w_i^\top x)\Bigr),
\]
with \(x\sim\mathcal{N}(0,I_d)\) and \(d \gg k_1,k_2\), our goal is to produce a hypothesis \(\widetilde F\) satisfying
\(\|\widetilde F - F\|_{L_2} \le \epsilon\), in time and with sample complexity polynomial in \(d,k_1,k_2,\epsilon^{-1}\).-->

---

## Repository Structure

```
.
├── HMD_mnist.py
├── HMD_fmnist.py
├── HMD_cifar10.py
├── HMR_mnist.py
├── HMR_fmnist.py
├── HMR_cifar10.py
├── mlp_benchmark.py
├── requirements.txt
└── slides/
    └── two_layer_relu_presentation.tex
    └── two_layer_relu_presentation.pdf
```

- **HMD_*.py**: run the Hierarchical Moment Decomposition pipeline on a given dataset.  
- **HMR_*.py**: run the Hermite Moment Regression pipeline on a given dataset.  
- **mlp_benchmark.py**: train and evaluate a standard multilayer perceptron baseline on MNIST, Fashion‑MNIST, or CIFAR‑10.  
- **slides/**: contains the LaTeX Beamer source and generated PDF for the project presentation.

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/two-layer-relu-learning.git
   cd two-layer-relu-learning
   ```

2. **Create a Python environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   _`requirements.txt` includes: `numpy`, `scipy`, `scikit-learn`, `torch`, `torchvision`, `tqdm`._

4. **(Optional) Build slides**  
   ```bash
   cd slides
   pdflatex two_layer_relu_presentation.tex
   # then open two_layer_relu_presentation.pdf
   ```

---

## Usage

### 1. Hierarchical Moment Decomposition (HMD)

- **MNIST**  
  ```bash
  python HMD_mnist.py --samples 60000 --k1 64 --k2 32 --output_dir outputs/hmd_mnist
  ```
- **Fashion‑MNIST**  
  ```bash
  python HMD_fmnist.py --samples 60000 --k1 64 --k2 32 --output_dir outputs/hmd_fmnist
  ```
- **CIFAR‑10**  
  ```bash
  python HMD_cifar10.py --samples 50000 --k1 64 --k2 32 --output_dir outputs/hmd_cifar10
  ```

### 2. Hermite Moment Regression (HMR)

- **MNIST**  
  ```bash
  python HMR_mnist.py --samples 60000 --k1 64 --degree 10 --output_dir outputs/hmr_mnist
  ```
- **Fashion‑MNIST**  
  ```bash
  python HMR_fmnist.py --samples 60000 --k1 64 --degree 10 --output_dir outputs/hmr_fmnist
  ```
- **CIFAR‑10**  
  ```bash
  python HMR_cifar10.py --samples 50000 --k1 64 --degree 10 --output_dir outputs/hmr_cifar10
  ```

### 3. MLP Benchmark

Train a standard MLP and compare accuracy/training time:

```bash
python mlp_benchmark.py --dataset mnist --hidden_sizes 128 64 --epochs 20 --batch_size 128
```

---

<!--## Methodology

### Hierarchical Moment Decomposition (HMD)

1. **Stage 1 – First‑Layer Subspace Recovery**  
   - Estimate Hermite moments up to order \(4k_1\):  
     \(\widehat M^m = \tfrac1N\sum_tF(x^{(t)})\,H_m(x^{(t)})\).  
   - Form \(Q_1(v)=\sum_{m=1}^{4k_1}\|\widehat M^m v\|^2\); recover subspace \(V_1\) via its top‑\(k_1\) eigenvectors.

2. **Stage 2 – Projected‑Space Learning**  
   - Project inputs: \(Z = \mathrm{Proj}_{V_1}(X)\sim N(0,I_{k_1})\).  
   - Approximate the second layer via low‑degree polynomial moments in \(Z\).  
   - Repeat eigen‑decomposition to recover \(\{a_{ji},b_j\}\).

3. **Hypothesis**  
   \[
   \widetilde F(x)
   = \sum_{j=1}^{k_2} b_j\,\mathrm{ReLU}\Bigl(\sum_{i=1}^{k_1} a_{ji}\,\mathrm{ReLU}(\hat w_i^\top x)\Bigr).
   \]

### Hermite Moment Regression (HMR)

1. **Stage 1 – Subspace Recovery**  
   - Compute \(T_m = \mathbb{E}[F(x)\,H_m(x)]\), build \(Q(v)=\sum_{m=1}^M \|T_m v\|^2\); extract top‑\(k_1\) eigenvectors \(V_1\).

2. **Stage 2 – Polynomial Approximation & Classification**  
   - Project \(z = V_1^\top x\).  
   - Expand \(G(z)\) in Hermite basis up to degree \(D=O(\epsilon^{-4/3})\).  
   - Fit a multinomial logistic regression on features \([H_1(z),…,H_D(z)]\).

---
-->
## Experimentation & Datasets

We evaluate on:

| Dataset           | Train / Test     | Input Size | Classes |
| ----------------- | ---------------- | ---------- | ------- |
| **MNIST**         | 60 000 / 10 000  | 28×28 gray | 10      |
| **Fashion‑MNIST** | 60 000 / 10 000  | 28×28 gray | 10      |
| **CIFAR‑10**      | 50 000 / 10 000  | 32×32 RGB  | 10      |

---

## Results

### Test Accuracy (%)

| Method                   | MNIST  | Fashion‑MNIST | CIFAR‑10 |
| ------------------------ | ------ | ------------- | -------- |
| **HMD**                  | 73.41  | 73.62         | 28.76    |
| **HMR**                  | 92.12  | 83.35         | 41.25    |
| **2‑hidden‑layer ReLU**  | **96.46** | **85.78**     | **47.22** |

### Training Time (seconds)

| Method                  | MNIST | Fashion‑MNIST | CIFAR‑10 |
| ----------------------- | ----- | ------------- | -------- |
| **HMD**                 | 5.63  | 5.37          | 3.31     |
| **HMR**                 | 1.70  | 1.63          | 0.96     |
| **2‑hidden‑layer ReLU** | 8.73  | 8.35          | 5.24     |

---

## Future Work & Limitations

- **Future Work:**  
  - Extend HMR to coreset‑based network compression.  
  - Relax Gaussian input assumption for real‑world data.

- **Limitations:**  
  - Assumes \(x\sim\mathcal{N}(0,I)\).  
  - Performance may degrade on non‑Gaussian inputs.

---

## References

1. Diakonikolas, I. & Kane, D. M. (2024). *Efficiently Learning One‑Hidden‑Layer ReLU Networks via Schur Polynomials*. **COLT 2024**, PMLR Vol. 247, 1–15.  
2. LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P. (1998). *Gradient‑Based Learning Applied to Document Recognition*. **IEEE Proc.**, 86(11), 2278–2324.  
3. Xiao, H., Rasul, K. & Vollgraf, R. (2017). *Fashion‑MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms*. **arXiv:1708.07747**.  
4. Krizhevsky, A. & Hinton, G. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto.  
5. Pedregosa, F. et al. (2011). *Scikit‑learn: Machine Learning in Python*. **JMLR**, 12, 2825–2830.  
