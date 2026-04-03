import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.env.warehouse_env import WarehouseEnvironment


def load_data(n_orders=15, items_per_order=(5, 15), seed=42):
    env = WarehouseEnvironment(width=200, height=200, depot=(0, 0))
    orders = env.generate_orders(num_orders=n_orders,
                                 items_per_order=items_per_order,
                                 seed=seed)
    points = env.flatten_orders(orders)
    depot = np.array(env.depot)
    print(f"[DATA] {len(points)} articles générés.")
    return points, depot

#algorithmes à comparer

def get_algorithms(n_clusters: int = 5, points: np.ndarray = None) -> dict:

    # Calibration automatique de eps pour DBSCAN
    # On estime eps = 15% de la diagonale de l'espace
    if points is not None:
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        auto_eps = 0.12 * np.sqrt(x_range**2 + y_range**2)
    else:
        auto_eps = 25

    algos = {
        "K-Means": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
        "GMM (EM)": GaussianMixture(n_components=n_clusters, random_state=42),
        "DBSCAN": DBSCAN(eps=auto_eps, min_samples=3),
    }
    return algos

# sélection de k optimal (K-Means, utilisé comme référence)


def find_optimal_k(points: np.ndarray, k_min=2, k_max=10) -> int:

    k_range = range(k_min, min(k_max + 1, len(points)))
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(points)
        scores.append(silhouette_score(points, labels))
    best_k = list(k_range)[int(np.argmax(scores))]
    print(f"[K-SELECT] k optimal = {best_k}  (silhouette={max(scores):.4f})")
    return best_k

#évaluation d'un clusturing

def evaluate_clustering(points: np.ndarray, labels: np.ndarray,
                         exec_time: float) -> dict:

    valid_mask = labels != -1  # -1 = bruit DBSCAN
    n_noise = int(np.sum(~valid_mask))
    noise_pct = round(100 * n_noise / len(labels), 1)
    n_clusters_found = len(set(labels[valid_mask]))

    if n_clusters_found < 2 or valid_mask.sum() < 10:
        # Clustering dégénéré (tout en bruit ou 1 seul cluster)
        return {
            "n_clusters": n_clusters_found,
            "noise_%": noise_pct,
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "exec_time_ms": round(exec_time * 1000, 2),
        }

    sil = silhouette_score(points[valid_mask], labels[valid_mask])
    db = davies_bouldin_score(points[valid_mask], labels[valid_mask])
    ch = calinski_harabasz_score(points[valid_mask], labels[valid_mask])

    return {
        "n_clusters": n_clusters_found,
        "noise_%": noise_pct,
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 2),
        "exec_time_ms": round(exec_time * 1000, 2),
    }

# comparaison

def compare_algorithms(points: np.ndarray, n_clusters: int) -> dict:
    algos = get_algorithms(n_clusters=n_clusters, points=points)
    results = {}

    print(f"\n{'='*55}")
    print(f"  COMPARAISON — {len(algos)} algorithmes sur {len(points)} points")
    print(f"{'='*55}")

    for name, algo in algos.items():
        t0 = time.perf_counter()

        if name == "GMM (EM)":
            algo.fit(points)
            labels = algo.predict(points)
        else:
            labels = algo.fit_predict(points)

        elapsed = time.perf_counter() - t0
        metrics = evaluate_clustering(points, labels, elapsed)
        results[name] = {"labels": labels, "metrics": metrics}

        sil_str = f"{metrics['silhouette']:.4f}" if metrics['silhouette'] is not None else "N/A"
        print(f"  [{name:15s}] k={metrics['n_clusters']:2d} | "
              f"Silhouette={sil_str} | "
              f"Temps={metrics['exec_time_ms']:.1f}ms | "
              f"Bruit={metrics['noise_%']}%")

    return results


# sélection du meilleur algorithme

def select_best(results: dict) -> str:

    valid = {
        name: res for name, res in results.items()
        if res["metrics"]["silhouette"] is not None
    }
    if not valid:
        print("[WARN] Aucun algorithme valide — DBSCAN tout en bruit ?")
        return list(results.keys())[0]

    best_name = max(valid, key=lambda n: (
        valid[n]["metrics"]["silhouette"],
        -valid[n]["metrics"]["exec_time_ms"]
    ))
    print(f"\n[BEST] → Algorithme sélectionné : {best_name} "
          f"(silhouette={valid[best_name]['metrics']['silhouette']})")
    return best_name



COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def _plot_single_clustering(ax, points, labels, title, depot):
    unique_labels = sorted(set(labels))
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = "gray" if lbl == -1 else COLORS[idx % len(COLORS)]
        label_name = "Bruit (DBSCAN)" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(points[mask, 0], points[mask, 1],
                   c=[color], s=30, alpha=0.75, label=label_name)
    ax.scatter(*depot, color='red', s=150, marker='*', zorder=10, label='Dépôt')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=6, loc='upper right')


def plot_all_clusterings(points: np.ndarray, results: dict,
                          depot: np.ndarray, best_name: str) -> None:
    #figure 1
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (name, res) in zip(axes, results.items()):
        labels = res["labels"]
        m = res["metrics"]
        sil_str = f"{m['silhouette']:.3f}" if m['silhouette'] else "N/A"
        star = " ★ BEST" if name == best_name else ""
        title = f"{name}{star}\nSilhouette={sil_str} | k={m['n_clusters']} | {m['exec_time_ms']}ms"
        _plot_single_clustering(ax, points, labels, title, depot)

    plt.suptitle("Comparaison des algorithmes de Clustering\n(Entrepôt — Order Picking)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("fig1_comparaison_clusterings.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVE] fig1_comparaison_clusterings.png")


def plot_metrics_comparison(results: dict) -> None:

    #Figure 2

    names = list(results.keys())
    metrics_to_plot = {
        "Silhouette Score\n(↑ mieux)": "silhouette",
        "Davies-Bouldin\n(↓ mieux)": "davies_bouldin",
        "Calinski-Harabasz\n(↑ mieux)": "calinski_harabasz",
        "Temps d'exécution (ms)\n(↓ mieux)": "exec_time_ms",
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    bar_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    for ax, (metric_label, metric_key), color in zip(axes, metrics_to_plot.items(), bar_colors):
        values = []
        for name in names:
            v = results[name]["metrics"].get(metric_key)
            values.append(v if v is not None else 0)

        bars = ax.bar(names, values, color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        ax.set_title(metric_label, fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # Valeur au-dessus de chaque barre
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:.2f}" if val else "N/A",
                    ha='center', va='bottom', fontsize=8)

    plt.suptitle("Comparaison des métriques par algorithme", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("fig2_metriques_comparaison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVE] fig2_metriques_comparaison.png")


def plot_best_clustering_detail(points: np.ndarray, labels: np.ndarray,
                                 depot: np.ndarray, best_name: str) -> None:

    # Figure 3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    unique_labels = sorted(set(l for l in labels if l != -1))
    centers = []

    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        pts = points[mask]
        color = COLORS[idx % len(COLORS)]
        ax1.scatter(pts[:, 0], pts[:, 1], c=[color], s=40, alpha=0.8)
        cx, cy = pts.mean(axis=0)
        centers.append((cx, cy))
        ax1.scatter(cx, cy, c=[color], s=200, edgecolors='black',
                    linewidths=1.5, marker='D', zorder=4)
        ax1.annotate(f"C{lbl}\n({mask.sum()})",
                     (cx, cy), textcoords="offset points", xytext=(6, 6),
                     fontsize=8, fontweight='bold')

    ax1.scatter(*depot, color='red', s=200, marker='*', zorder=10, label='Dépôt')
    ax1.set_title(f"Meilleur algorithme : {best_name}\n(centroïdes = ◆)",
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel("X"); ax1.set_ylabel("Y")
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend()

    # Distribution des tailles de clusters
    sizes = [np.sum(labels == lbl) for lbl in unique_labels]
    ax2.bar([f"C{lbl}" for lbl in unique_labels], sizes,
            color=[COLORS[i % len(COLORS)] for i in range(len(unique_labels))],
            edgecolor='black', linewidth=0.6)
    ax2.set_title("Taille de chaque cluster\n(équilibre de la charge)", fontsize=11)
    ax2.set_xlabel("Cluster"); ax2.set_ylabel("Nombre d'articles")
    ax2.axhline(y=np.mean(sizes), color='red', linestyle='--',
                label=f"Moyenne = {np.mean(sizes):.1f}")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    plt.suptitle(f"Analyse détaillée — {best_name}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("fig3_best_clustering_detail.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVE] fig3_best_clustering_detail.png")


# tableau récapitulatif

def print_summary_table(results: dict, best_name: str) -> pd.DataFrame:

    rows = []
    for name, res in results.items():
        m = res["metrics"]
        rows.append({
            "Algorithme": ("★ " if name == best_name else "  ") + name,
            "k trouvé": m["n_clusters"],
            "Silhouette ↑": m["silhouette"] if m["silhouette"] else "N/A",
            "Davies-Bouldin ↓": m["davies_bouldin"] if m["davies_bouldin"] else "N/A",
            "Calinski-Harabasz ↑": m["calinski_harabasz"] if m["calinski_harabasz"] else "N/A",
            "Bruit (%)": m["noise_%"],
            "Temps (ms)": m["exec_time_ms"],
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("  TABLEAU RÉCAPITULATIF — COMPARAISON DES ALGORITHMES ML")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print("  Légende : ↑ = plus haut est mieux | ↓ = plus bas est mieux | ★ = BEST")
    print("=" * 80)


    df.to_csv("tableau_comparaison_ml.csv", index=False)
    print("[SAVE] tableau_comparaison_ml.csv")

    return df

# export vers la métaheuristique


def export_for_metaheuristic(points: np.ndarray, labels: np.ndarray,
                              depot: np.ndarray) -> dict:

    valid_labels = sorted(set(l for l in labels if l != -1))
    batches = {}
    centers = []

    for lbl in valid_labels:
        mask = labels == lbl
        batch_pts = points[mask]
        batches[lbl] = batch_pts
        centers.append(batch_pts.mean(axis=0))

    centers = np.array(centers)

    # Matrice de distances Manhattan entre centroïdes
    n = len(centers)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = abs(centers[i, 0] - centers[j, 0]) + \
                                  abs(centers[i, 1] - centers[j, 1])

    payload = {
        "batches": batches,
        "centers": centers,
        "depot": depot,
        "n_clusters": len(valid_labels),
        "dist_matrix": dist_matrix,
    }

    print(f"\n[EXPORT] {payload['n_clusters']} batches prêts pour ACO/AG")
    print(f"[EXPORT] Centroïdes :\n{centers.round(1)}")
    print(f"[EXPORT] Matrice distances (Manhattan) :\n{dist_matrix.round(1)}")

    return payload



# pipeline


def run_full_pipeline(n_orders=15, items_per_order=(5, 15), seed=42):

    print("\n" + "█" * 60)
    print("  PIPELINE ML — COMPARAISON ALGORITHMES DE CLUSTERING")
    print("  Projet Optimisation Combinatoire — M1 MIAGE MIXTE")
    print("█" * 60)

    #  données
    points, depot = load_data(n_orders=n_orders,
                               items_per_order=items_per_order,
                               seed=seed)

    # k optimal
    print("\n[STEP 1] Recherche du k optimal...")
    best_k = find_optimal_k(points, k_min=2, k_max=8)

    # comparaison
    print("\n[STEP 2] Comparaison des algorithmes...")
    results = compare_algorithms(points, n_clusters=best_k)

    # sélection du meilleur
    print("\n[STEP 3] Sélection du meilleur algorithme...")
    best_name = select_best(results)

    # Tableau récapitulatif
    print("\n[STEP 4] Tableau récapitulatif...")
    df = print_summary_table(results, best_name)

    # figure  tous les clusterings
    print("\n[STEP 5] Génération des figures...")
    plot_all_clusterings(points, results, depot, best_name)

    # figure 2  métriques en barres
    plot_metrics_comparison(results)

    # figure 3  détail du meilleur
    best_labels = results[best_name]["labels"]
    plot_best_clustering_detail(points, best_labels, depot, best_name)

    # export
    print("\n[STEP 6] Export pour la métaheuristique...")
    payload = export_for_metaheuristic(points, best_labels, depot)

    print("\n" + "█" * 60)
    print(f"  PIPELINE TERMINÉ")
    print(f"  Meilleur algo : {best_name}")
    print(f"  Fichiers sauvegardés :")
    print(f"    - fig1_comparaison_clusterings.png")
    print(f"    - fig2_metriques_comparaison.png")
    print(f"    - fig3_best_clustering_detail.png")
    print(f"    - tableau_comparaison_ml.csv")
    print("█" * 60)

    return payload, best_name, df



if __name__ == "__main__":
    payload, best_name, df = run_full_pipeline(
        n_orders=15,
        items_per_order=(5, 15),
        seed=42
    )