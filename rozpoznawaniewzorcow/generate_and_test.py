import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score

# ==============================================================================
# 1. USTALENIE PROTOKOŁU I ZIARNA LOSOWOŚCI (Kontrola poprawności)
# ==============================================================================
SEED = 42
K_NEIGHBORS = 20
METRIC = "euclidean"


def generate_stage1_data(seed=SEED):
    """
    Generuje zbiór danych inspirowany Rysunkiem 1 z publikacji Breunig et al. (2000):
    - Skupisko C1: gęste (małe odchylenie standardowe)
    - Skupisko C2: rzadkie (duże odchylenie standardowe)
    - 20 oznaczonych anomalii (w tym anomalie lokalne blisko C1 oraz globalne)
    Łącznie: 470 obiektów (limit zadania: do 500 obiektów).
    """
    rng = np.random.default_rng(seed)

    # C1: Gęste skupisko (250 punktów wokół [-5, -5], std = 0.4)
    X_dense, _ = make_blobs(
        n_samples=250, centers=[[-5.0, -5.0]], cluster_std=0.4, random_state=seed
    )

    # C2: Rzadkie skupisko (200 punktów wokół [4, 4], std = 2.2)
    X_sparse, _ = make_blobs(
        n_samples=200, centers=[[4.0, 4.0]], cluster_std=2.2, random_state=seed + 1
    )

    # Anomalie (20 punktów) - wariant bazowy: rozproszone / odosobnione
    # 5 silnych anomalii lokalnych (tuż obok gęstego klastra C1 - w odległości ~1.5-2.0,
    # która w klastrze C2 byłaby zupełnie normalna!)
    local_outliers = np.array(
        [[-3.2, -5.0], [-5.0, -3.1], [-6.8, -5.0], [-5.0, -6.9], [-3.6, -3.6]]
    )

    # 15 anomalii globalnych rozrzuconych w przestrzeni
    global_outliers = rng.uniform(low=-11.0, high=11.0, size=(15, 2))

    X_outliers = np.vstack([local_outliers, global_outliers])

    # Łączenie danych
    X = np.vstack([X_dense, X_sparse, X_outliers])

    # Etykiety odniesienia: 0 = obiekt normalny (inlier), 1 = anomalia (outlier)
    y = np.hstack(
        [
            np.zeros(len(X_dense) + len(X_sparse), dtype=int),
            np.ones(len(X_outliers), dtype=int),
        ]
    )

    # Informacja pomocnicza o grupie (do analizy i wizualizacji)
    group = (
        ["C1_dense"] * len(X_dense)
        + ["C2_sparse"] * len(X_sparse)
        + ["Outlier_local"] * len(local_outliers)
        + ["Outlier_global"] * len(global_outliers)
    )

    df = pd.DataFrame(
        {"x1": X[:, 0], "x2": X[:, 1], "is_anomaly": y, "group": group}
    )

    return X, y, df


def verify_manual_example():
    """
    Weryfikacja przykładu teoretycznego z opisu zadania (Etap 1):
    Punkt o lrd_p = 2, dwóch sąsiadów o lrd_o1 = 4, lrd_o2 = 4.
    """
    lrd_p = 2.0
    lrd_neighbors = np.array([4.0, 4.0])
    lof_p = np.mean(lrd_neighbors / lrd_p)
    print("=== WERYFIKACJA PRZYKŁADU TEORETYCZNEGO (ETAP 1) ===")
    print(f"Dla lrd(p) = {lrd_p} oraz sąsiadów z lrd = {lrd_neighbors.tolist()}:")
    print(f"Wyliczony LOF(p) = {lof_p:.2f}\n")


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # 1. Przykład teoretyczny
    verify_manual_example()

    # 2. Generowanie i zapis zbioru danych
    X, y, df = generate_stage1_data()
    csv_path = os.path.join("data", "synthetic_lof_stage1.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wygenerowano zbiór danych: {len(df)} obiektów (w tym {y.sum()} anomalii).")
    print(f"Zapisano do pliku: {csv_path}\n")

    # 3. Mała próba uruchomienia LOF (Kontrola poprawności kierunku wyniku!)
    lof = LocalOutlierFactor(n_neighbors=K_NEIGHBORS, metric=METRIC)
    lof.fit(X)

    # UWAGA METODOLOGICZNA (wymagana w zadaniu):
    # scikit-learn w atrybucie negative_outlier_factor_ zwraca wartości UJEMNE (-LOF).
    # Im mniejsza (bardziej ujemna) wartość, tym większa anomalia.
    # Aby uzyskać oryginalny dodatni współczynnik LOF zgodny z publikacją, mnożymy przez -1:
    lof_scores = -lof.negative_outlier_factor_
    df["lof_score"] = lof_scores

    # Wyliczenie wymaganej w protokole metryki Average Precision (AP)
    ap_score = average_precision_score(y, lof_scores)

    print("=== WSTĘPNA PRÓBA URUCHOMIENIA (k=20, metryka euklidesowa) ===")
    print(f"Średni LOF dla gęstego skupiska (C1): {df[df['group']=='C1_dense']['lof_score'].mean():.2f}")
    print(f"Średni LOF dla rzadkiego skupiska (C2): {df[df['group']=='C2_sparse']['lof_score'].mean():.2f}")
    print(f"Średni LOF dla anomalii lokalnych:    {df[df['group']=='Outlier_local']['lof_score'].mean():.2f}")
    print(f"Średni LOF dla anomalii globalnych:   {df[df['group']=='Outlier_global']['lof_score'].mean():.2f}")
    print(f"Average Precision (AP) rankingu LOF:  {ap_score:.4f}")

    # 4. Wizualizacja do slajdów i protokołu
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Wykres 1: Podział na grupy i wygenerowane anomalie
    ax1 = axes[0]
    c1 = df[df["group"] == "C1_dense"]
    c2 = df[df["group"] == "C2_sparse"]
    out_loc = df[df["group"] == "Outlier_local"]
    out_glob = df[df["group"] == "Outlier_global"]

    ax1.scatter(c1["x1"], c1["x2"], c="#1f77b4", s=20, alpha=0.7, label="C1: Skupisko gęste (n=250, std=0.4)")
    ax1.scatter(c2["x1"], c2["x2"], c="#2ca02c", s=20, alpha=0.7, label="C2: Skupisko rzadkie (n=200, std=2.2)")
    ax1.scatter(out_loc["x1"], out_loc["x2"], c="#ff7f0e", marker="X", s=90, edgecolor="black", label="Anomalie lokalne (n=5)")
    ax1.scatter(out_glob["x1"], out_glob["x2"], c="#d62728", marker="^", s=80, edgecolor="black", label="Anomalie globalne (n=15)")
    ax1.set_title("Wygenerowany zbiór danych (N=470, seed=42)")
    ax1.set_xlabel("Cecha $x_1$")
    ax1.set_ylabel("Cecha $x_2$")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Wykres 2: Wizualizacja wartości LOF (rozmiar okręgu proporcjonalny do LOF)
    ax2 = axes[1]
    ax2.scatter(X[:, 0], X[:, 1], c="black", s=10, alpha=0.6, label="Obiekty danych")
    # Skalowanie promienia okręgów dla czytelności
    radius = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())
    ax2.scatter(
        X[:, 0], X[:, 1], s=1200 * radius, edgecolors="#d62728",
        facecolors="none", linewidths=1.2, label="Wynik LOF (promień $\propto$ LOF)"
    )
    ax2.set_title(f"Wyniki algorytmu LOF (k={K_NEIGHBORS}, AP={ap_score:.3f})")
    ax2.set_xlabel("Cecha $x_1$")
    ax2.set_ylabel("Cecha $x_2$")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join("plots", "stage1_dataset_preview.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Wykres zapisano do: {plot_path}")


if __name__ == "__main__":
    main()