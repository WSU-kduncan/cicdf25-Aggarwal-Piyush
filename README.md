# K-Means Clustering on Wine Quality Dataset
## Introduction

In this project, I built the K-Means clustering algorithm from scratch and used it on the red and white wine quality datasets. The goal was to group wines based on their chemical features and see how well the clusters matched real wine types and quality scores.

## Choosing the Number of Clusters
I used the Elbow Method to find the best value of k. The graph showed that after k = 4 or 5, the drop in error slowed down. So, I chose k = 5 as a good balance between accuracy and simplicity.

## Results

- Two ways were tested to start the cluster centers:

  - Strategy 1: Random values between the smallest and biggest feature values.

  - Strategy 2: Random values within the middle range (the interquartile range), which avoids extreme outliers.

- At k = 5, here are the results:

  - SSE: 42,720 (Strategy 1) vs 38,069 (Strategy 2)

  - Purity vs Quality: 0.449 (S1) and 0.454 (S2)

  - Purity vs Color: 0.982 (S1) and 0.976 (S2)

Strategy 2 gave better and more stable clusters. Both methods mainly separated wines by color (red or white), not by quality, which makes sense because many wines with different quality scores have similar chemical properties.

## PCA Visualization

I used PCA to reduce the data to two dimensions for visualization. The first two components explained about 50% of the data’s variation.
The plots showed clear groups separating red and white wines, and some smaller subgroups within each color. Strategy 2 looked a bit cleaner and more balanced.

## Conclusion

The K-Means algorithm successfully grouped wines by their chemical similarities.
It worked better for color separation than for quality prediction.
Overall, Strategy 2 (IQR-based initialization) gave the best results with lower SSE and smoother clusters.


``` # kmeans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def zscore(X):
    m = X.mean(0, keepdims=True)
    s = X.std(0, keepdims=True)
    s[s == 0] = 1.0
    return (X - m) / s, m, s

def pca2(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T, (S**2)/(len(X)-1)

def purity(y_true, y_pred):
    yt = pd.factorize(y_true)[0]
    yp = pd.factorize(y_pred)[0]
    k = yp.max() + 1
    total = len(y_true)
    good = 0
    for c in range(k):
        idx = (yp == c)
        if idx.any():
            vals, counts = np.unique(yt[idx], return_counts=True)
            good += counts.max()
    return good / total


def init_s1(X, k, rng):
    lo = X.min(0); hi = X.max(0)
    return rng.uniform(lo, hi, size=(k, X.shape[1]))

def init_s2(X, k, rng):
    C = np.empty((k, X.shape[1]))
    for j in range(X.shape[1]):
        col = np.sort(X[:, j])
        q1 = int(0.25*(len(col)-1))
        q3 = int(0.75*(len(col)-1))
        mid = col[q1:q3+1]
        lo, hi = mid.min(), mid.max()
        if lo == hi:
            lo, hi = col.min(), col.max()
        C[:, j] = rng.uniform(lo, hi, size=k)
    return C


# k-means
def kmeans(X, k, init="s1", max_iter=200, tol=1e-4, seed=42):
    rng = np.random.RandomState(seed)
    if init == "s1":
        C = init_s1(X, k, rng)
    else:
        C = init_s2(X, k, rng)

    for _ in range(max_iter):
        d2 = ((X[:, None, :] - C[None, :, :])**2).sum(axis=2)
        lab = d2.argmin(1)
        C_new = C.copy()
        for c in range(k):
            mask = (lab == c)
            if mask.any():
                C_new[c] = X[mask].mean(0)
            else:
                C_new[c] = X[rng.randint(0, len(X))]
        shift = np.linalg.norm(C_new - C)
        C = C_new
        if shift <= tol:
            break
    inertia = ((X - C[lab])**2).sum()
    return lab, C, inertia


def elbow_plot(X, ks, seed=42):
    sse = []
    for k in ks:
        _, _, s = kmeans(X, k, init="s1", seed=seed)
        sse.append(s)
    plt.figure(figsize=(6,4))
    plt.plot(ks, sse, marker="o")
    plt.title("Elbow (init: strategy 1)")
    plt.xlabel("k"); plt.ylabel("SSE")
    plt.tight_layout()
    plt.show()
    return np.array(sse)



def main():
    red = pd.read_csv("D:/SCHOOL/Fall 25/CS 4850 AI/Assignment3/winequality-red.csv", sep=";")
    white = pd.read_csv("D:/SCHOOL/Fall 25/CS 4850 AI/Assignment3/winequality-white.csv", sep=";")
    red["color"] = "red";  white["color"] = "white"
    df = pd.concat([red, white], ignore_index=True)

    feats = [c for c in df.columns if c not in ("quality","color")]
    Xraw = df[feats].to_numpy(dtype=float)
    y_quality = df["quality"].to_numpy()
    y_color   = df["color"].to_numpy()

    X, mu, st = zscore(Xraw)


    ks = list(range(2, 11))
    sse = elbow_plot(X, ks)
    print("SSE by k:", dict(zip(ks, map(float, sse))))

    k = 5

    lab1, C1, s1 = kmeans(X, k, init="s1", seed=42)
    lab2, C2, s2 = kmeans(X, k, init="s2", seed=42)

    p_q1 = purity(y_quality, lab1)
    p_q2 = purity(y_quality, lab2)
    p_c1 = purity(y_color,   lab1)
    p_c2 = purity(y_color,   lab2)
    print(f"k={k}  SSE(s1)={s1:.0f}  SSE(s2)={s2:.0f}")
    print(f"purity vs quality: s1={p_q1:.3f}  s2={p_q2:.3f}")
    print(f"purity vs color:   s1={p_c1:.3f}  s2={p_c2:.3f}")

    X2, var = pca2(X)
    evr2 = var[:2].sum() / var.sum()


    plt.figure(figsize=(6,5))
    plt.scatter(X2[:,0], X2[:,1], s=8, c=lab1)
    plt.title(f"PCA 2D • strategy 1 • k={k} • EVR={evr2:.3f}")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,5))
    plt.scatter(X2[:,0], X2[:,1], s=8, c=lab2)
    plt.title(f"PCA 2D • strategy 2 • k={k} • EVR={evr2:.3f}")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,5))
    plt.scatter(X2[:,0], X2[:,1], s=8, c=y_quality)
    plt.title("PCA 2D • true quality (reference)")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
```


# K-Means Clustering on Cancer Gene Expression Data (Part 2)

## Goal:
The purpose of this project was to explore how K-Means can group cancer samples based on their gene expression patterns. The dataset includes thousands of genes from five different types of tumors. I also used PCA (Principal Component Analysis) to simplify the data while keeping most of the important information.

## Process:

- I loaded the dataset and normalized it so that very large or small values wouldn’t affect the results.

- I ran K-Means with k = 5 to form five clusters, one for each tumor type.

- Then I used PCA to reduce the number of features, keeping 99% of the total variance.

- Finally, I ran K-Means again on the PCA data and compared both results.

- I created 2D plots to visualize how the samples were grouped before and after PCA.

## Results:

In the original dataset, the K-Means model achieved an inertia value of about 12.2 million and a purity score of 0.006. 

After applying PCA, the model’s inertia slightly decreased to around 12.05 million, and the purity remained the same at 0.006.
Even though the purity didn’t change, PCA made the data much smaller and easier to process. The clusters also looked a bit more separated when plotted in two dimensions. Both runs gave a purity of 0.006, meaning the clusters were not perfectly matched to the actual tumor types, but PCA still helped make the data easier to analyze and visualize. The PCA version used 725 components instead of thousands of original features, making it much faster to process.

## Discussion:
PCA helped reduce the dataset’s complexity while keeping almost all the information. After PCA, the clusters looked slightly more separated and clearer in 2D graphs. This shows that PCA can make high-dimensional data more understandable without losing accuracy.

However, one downside is that PCA combines many genes into abstract “components,” so it’s harder to tell which specific genes cause the clustering differences.

## Conclusion:
Using PCA with K-Means made the results simpler, faster, and easier to visualize. The clustering accuracy (purity) stayed the same, but the PCA-transformed data was more organized and efficient. Overall, PCA improved the structure and clarity of the analysis without sacrificing performance.

```
# p2kmeans.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.errors import DtypeWarning
import warnings

warnings.simplefilter("ignore", DtypeWarning)


def load_numeric_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, header=None, low_memory=False)
    # Coerce all to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def clean_matrix(df: pd.DataFrame) -> np.ndarray:


    # Drop all-NaN columns
    df = df.loc[:, ~df.isna().all()]

    if df.shape[1] == 0:
        raise ValueError("All columns are NaN. Check the input file formatting.")


    med = df.median(numeric_only=True)
    df = df.fillna(med)

    X = df.to_numpy(dtype=float)

    # Replace
    nonfinite_mask = ~np.isfinite(X)
    if nonfinite_mask.any():
        X[nonfinite_mask] = np.nan
        col_med = np.nanmedian(X, axis=0)
     
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        idxs = np.where(np.isnan(X))
        X[idxs] = np.take(col_med, idxs[1])


    X = np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def standardize(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    Z = (X - mean) / std

    Z = np.nan_to_num(Z, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return Z



def purity_score(y_true, y_pred):
    y_true = pd.factorize(y_true)[0]
    y_pred = pd.factorize(y_pred)[0]
    total, correct = len(y_true), 0
    for c in np.unique(y_pred):
        mask = (y_pred == c)
        vals, counts = np.unique(y_true[mask], return_counts=True)
        if len(counts): correct += counts.max()
    return correct / total if total else 0.0



def pca(X, keep_var=0.99):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S**2) / (len(X) - 1)
    cum = np.cumsum(var) / var.sum() if var.sum() > 0 else np.ones_like(var)
    r = np.searchsorted(cum, keep_var) + 1
    r = max(1, min(r, Vt.shape[0]))
    print(f"PCA kept {r} components ({cum[r-1]*100:.2f}% variance)")
    return Xc @ Vt[:r].T

def to2D(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

def to3D(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:3].T

# K-Means

def _kmeanspp_init(X, k, rng):
    n = X.shape[0]

    centers = np.empty((k, X.shape[1]), dtype=float)
    first = rng.randint(0, n)
    centers[0] = X[first]

    d2 = np.full(n, np.inf)
    for i in range(1, k):

        diff = X - centers[i-1]
        d2 = np.minimum(d2, np.einsum("ij,ij->i", diff, diff))
    
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0/n)
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
    return centers

def kmeans(X, k=5, max_iter=200, tol=1e-4, seed=42):
    rng = np.random.RandomState(seed)
    n, d = X.shape


    centers = _kmeanspp_init(X, k, rng)

    prev_inertia = None
    for _ in range(max_iter):

        dist = ((X[:, None, :] - centers[None, :, :])**2).sum(2)
        labels = dist.argmin(1)

        new_centers = centers.copy()
        for i in range(k):
            pts = X[labels == i]
            if len(pts):
                new_centers[i] = pts.mean(0)
            else:
                new_centers[i] = X[rng.randint(0, n)]

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers

        inertia = ((X - centers[labels])**2).sum()
        if prev_inertia is not None and abs(prev_inertia - inertia) < tol:
            break
        prev_inertia = inertia

    sse = ((X - centers[labels])**2).sum()
    return labels, sse



def main():
    # data
    X_df = load_numeric_csv("datap2.csv")
    X = clean_matrix(X_df)
    X = standardize(X)

    y = pd.read_csv("labelsp2.csv", header=None).iloc[:, 0].to_numpy()

    print("Running K-Means on raw data (k=5)...")
    labels_raw, sse_raw = kmeans(X, k=5, seed=42)
    purity_raw = purity_score(y, labels_raw)
    print(f"SSE (raw): {sse_raw:.0f}")
    print(f"Purity (raw): {purity_raw:.3f}")

    print("\nApplying PCA (retain 99% variance)...")
    Xp = pca(X, keep_var=0.99)

    print("Running K-Means on PCA-reduced data (k=5)...")
    labels_pca, sse_pca = kmeans(Xp, k=5, seed=42)
    purity_pca = purity_score(y, labels_pca)
    print(f"SSE (PCA): {sse_pca:.0f}")
    print(f"Purity (PCA): {purity_pca:.3f}")

    # 2D
    X2_raw = to2D(X)
    X2_pca = to2D(Xp)

    plt.figure(figsize=(6,5))
    plt.scatter(X2_raw[:,0], X2_raw[:,1], c=labels_raw, s=10)
    plt.title("K-Means on Raw Data (2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,5))
    plt.scatter(X2_pca[:,0], X2_pca[:,1], c=labels_pca, s=10)
    plt.title("K-Means on PCA Data (2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.show()

    # 3D
    X3_raw = to3D(X)
    X3_pca = to3D(Xp)

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X3_raw[:,0], X3_raw[:,1], X3_raw[:,2], c=labels_raw, s=8)
    ax.set_title("K-Means on Raw Data (3D)")
    plt.show()

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X3_pca[:,0], X3_pca[:,1], X3_pca[:,2], c=labels_pca, s=8)
    ax.set_title("K-Means on PCA Data (3D)")
    plt.show()

    print("\nSummary:")
    print(f"Purity (Raw): {purity_raw:.3f}")
    print(f"Purity (PCA): {purity_pca:.3f}")

if __name__ == "__main__":
    main()
```#   c i c d f 2 5 - A g g a r w a l - P i y u s h  
 