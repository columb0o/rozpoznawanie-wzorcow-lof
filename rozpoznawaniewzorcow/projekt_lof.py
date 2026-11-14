import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score

# ==============================================================================
# PARAMETRY
# ==============================================================================
SEED = 42
K_NEIGHBORS = 20
METRIC = "euclidean"

def verify_manual_example():
    """Weryfikacja przykładu teoretycznego z opisu zadania (Etap 1)"""
    lrd_p = 2.0
    lrd_neighbors = np.array([4.0, 4.0])
    lof_p = np.mean(lrd_neighbors / lrd_p)
    print("=== WERYFIKACJA PRZYKŁADU TEORETYCZNEGO (ETAP 1) ===")
    print(f"Dla lrd(p) = {lrd_p} oraz sąsiadów z lrd = {lrd_neighbors.tolist()}:")
    print(f"Wyliczony LOF(p) = {lof_p:.2f}\n")

def generate_data(seed=SEED):
    """Generator z poprawką nachodzenia anomalii na dane normalne (Etap 2)"""
    rng = np.random.default_rng(seed)

    # C1: Gęste skupisko
    X_dense, _ = make_blobs(n_samples=250, centers=[[-5.0, -5.0]], cluster_std=0.4, random_state=seed)

    # C2: Rzadkie skupisko
    X_sparse, _ = make_blobs(n_samples=200, centers=[[4.0, 4.0]], cluster_std=2.2, random_state=seed + 1)

    # Anomalie lokalne (bezpiecznie odsunięte od C1)
    local_outliers = np.array([
        [-2.8, -5.0], [-5.0, -2.8], [-7.2, -5.0], [-5.0, -7.2], [-3.4, -3.4]
    ])

    # Anomalie globalne (rejection sampling - odrzucanie wpadających w klastry)
    global_outliers = []
    while len(global_outliers) < 15:
        pt = rng.uniform(low=-11.0, high=11.0, size=2)
        dist_to_c1 = np.linalg.norm(pt - np.array([-5.0, -5.0]))
        dist_to_c2 = np.linalg.norm(pt - np.array([4.0, 4.0]))
        if dist_to_c1 > 3.0 and dist_to_c2 > 7.5:
            global_outliers.append(pt)
    global_outliers = np.array(global_outliers)

    X_outliers = np.vstack([local_outliers, global_outliers])
    X = np.vstack([X_dense, X_sparse, X_outliers])

    y = np.hstack([
        np.zeros(len(X_dense) + len(X_sparse), dtype=int),
        np.ones(len(X_outliers), dtype=int),
    ])

    group = (
        ["C1_dense"] * len(X_dense)
        + ["C2_sparse"] * len(X_sparse)
        + ["Outlier_local"] * len(local_outliers)
        + ["Outlier_global"] * len(global_outliers)
    )

    return pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "is_anomaly": y, "group": group})

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Weryfikacja ręczna (Etap 1)
    verify_manual_example()

    # 2. Generowanie i zapis danych (Etap 1 & 2)
    df = generate_data()
    csv_path = os.path.join(data_dir, "synthetic_lof_dataset.csv")
    df.to_csv(csv_path, index=False)
    
    X = df[["x1", "x2"]].values
    y = df["is_anomaly"].values

    # 3. Obliczanie miary lokalnej LOF (Etap 2)
    lof = LocalOutlierFactor(n_neighbors=K_NEIGHBORS, metric=METRIC)
    lof.fit(X)
    df["lof_score"] = -lof.negative_outlier_factor_
    ap_lof = average_precision_score(y, df["lof_score"])

    # 4. Obliczanie Miary Globalnej (Etap 2)
    global_center = X.mean(axis=0)
    df["global_score"] = np.linalg.norm(X - global_center, axis=1)
    ap_global = average_precision_score(y, df["global_score"])

    print("=== PORÓWNANIE METRYK (Etap 2) ===")
    print(f"Average Precision - LOF:              {ap_lof:.4f}")
    print(f"Average Precision - Miara Globalna:   {ap_global:.4f}\n")

    # 5. Weryfikacja praktyczna
    p_idx = 450 # Pierwszy lokalny outlier
    p_lof = df.iloc[p_idx]["lof_score"]
    print("=== WERYFIKACJA PRAKTYCZNA (Odniesienie do teorii) ===")
    print(f"Badany punkt (Anomalia lokalna): {X[p_idx]}")
    print(f"Wynik LOF z biblioteki scikit-learn: {p_lof:.2f}")
    print("Zgodnie z wyliczeniami z Etapu 1, wysoki współczynnik wynika")
    print("z gęstego otoczenia (wysokie lrd sąsiadów) przy rzadkim lrd punktu.\n")

    # 6. Wizualizacja Porównawcza (Tylko TOP 20 anomalii i mniejsze znaczniki)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    out_loc = df[df["group"] == "Outlier_local"]
    out_glob = df[df["group"] == "Outlier_global"]
    normal_data = df[df["is_anomaly"] == 0]

    # Zmniejszone rozmiary znaczników prawdziwych anomalii (z 90/80 na 60/50)
    MARKER_SIZE_LOC = 60
    MARKER_SIZE_GLOB = 50

    # ==========================================
    # Miara Globalna
    # ==========================================
    ax1 = axes[0]
    ax1.scatter(normal_data["x1"], normal_data["x2"], c="black", s=10, alpha=0.5, label="Dane normalne")
    ax1.scatter(out_loc["x1"], out_loc["x2"], c="#ff7f0e", marker="X", s=MARKER_SIZE_LOC, edgecolor="black", label="Anomalie lokalne")
    ax1.scatter(out_glob["x1"], out_glob["x2"], c="#d62728", marker="^", s=MARKER_SIZE_GLOB, edgecolor="black", label="Anomalie globalne")
    
    # Wyciągamy 20 punktów z najwyższym wynikiem miary globalnej
    top_20_glob_idx = df["global_score"].nlargest(20).index
    rad_glob = (df["global_score"] - df["global_score"].min()) / (df["global_score"].max() - df["global_score"].min())
    
    # Rysujemy okręgi TYLKO dla top 20 punktów
    ax1.scatter(X[top_20_glob_idx, 0], X[top_20_glob_idx, 1], 
                s=300 * rad_glob[top_20_glob_idx], edgecolors="#1f77b4", facecolors="none")
    
    ax1.scatter(global_center[0], global_center[1], c="red", marker="P", s=150, label="Globalne Centrum")
    ax1.scatter([], [], s=150, edgecolors="#1f77b4", facecolors="none", label="Wynik (Tylko Top 20)")

    ax1.set_title(f"Miara Globalna (AP={ap_global:.3f})\nAnomalie lokalne niewykryte (poza TOP 20)")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_xlim(-13, 13)
    ax1.set_ylim(-13, 13)

    # ==========================================
    # Miara Lokalna (LOF)
    # ==========================================
    ax2 = axes[1]
    ax2.scatter(normal_data["x1"], normal_data["x2"], c="black", s=10, alpha=0.5, label="Dane normalne")
    ax2.scatter(out_loc["x1"], out_loc["x2"], c="#ff7f0e", marker="X", s=MARKER_SIZE_LOC, edgecolor="black", label="Anomalie lokalne")
    ax2.scatter(out_glob["x1"], out_glob["x2"], c="#d62728", marker="^", s=MARKER_SIZE_GLOB, edgecolor="black", label="Anomalie globalne")
    
    # Wyciągamy 20 punktów z najwyższym wynikiem LOF
    top_20_lof_idx = df["lof_score"].nlargest(20).index
    rad_lof = (df["lof_score"] - df["lof_score"].min()) / (df["lof_score"].max() - df["lof_score"].min())
    
    # Rysujemy okręgi TYLKO dla top 20 punktów
    ax2.scatter(X[top_20_lof_idx, 0], X[top_20_lof_idx, 1], 
                s=300 * rad_lof[top_20_lof_idx], edgecolors="#d62728", facecolors="none")
    
    ax2.scatter([], [], s=150, edgecolors="#d62728", facecolors="none", label="Wynik LOF (Tylko Top 20)")

    ax2.set_title(f"Miara Lokalna LOF (AP={ap_lof:.3f})\nAnomalie lokalne wykryte poprawnie (w TOP 20)")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_xlim(-13, 13)
    ax2.set_ylim(-13, 13)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "lof_vs_global_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Zapisano wizualizację do: {plot_path}")
    print(f"Zapisano zbiór danych do: {csv_path}")

if __name__ == "__main__":
    main()