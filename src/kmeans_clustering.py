
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from env.warehouse_env import WarehouseEnvironment


# =============================================================================
# PARTIE 1 : DÉTERMINATION DU NOMBRE OPTIMAL DE CLUSTERS (K)
# Méthode du coude (Elbow Method) + Silhouette Score
# =============================================================================

def find_optimal_k(data_points, k_min=2, k_max=10):
    """
    Détermine le nombre optimal de clusters K via :
    - La méthode du coude (inertie / WCSS)
    - Le silhouette score (mesure de cohésion et séparation des clusters)

    :param data_points: Tableau numpy des coordonnées (x, y) de tous les articles
    :param k_min: Nombre minimum de clusters à tester
    :param k_max: Nombre maximum de clusters à tester
    :return: Le K optimal selon le silhouette score
    """
    inertias = []
    silhouette_scores = []
    k_values = range(k_min, k_max + 1)

    print("=== Recherche du K optimal ===")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_points)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(data_points, labels)
        silhouette_scores.append(score)
        print(f"  K={k} | Inertie={kmeans.inertia_:.2f} | Silhouette={score:.4f}")

    # --- Graphique Méthode du coude ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(k_values, inertias, marker='o', color='steelblue', linewidth=2)
    axes[0].set_title("Méthode du Coude (Elbow Method)", fontsize=13)
    axes[0].set_xlabel("Nombre de clusters K")
    axes[0].set_ylabel("Inertie (WCSS)")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(k_values, silhouette_scores, marker='s', color='darkorange', linewidth=2)
    axes[1].set_title("Silhouette Score par K", fontsize=13)
    axes[1].set_xlabel("Nombre de clusters K")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Analyse pour le choix du K optimal", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("optimal_k_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()

    # K optimal = celui avec le silhouette score maximal
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"\n>>> K optimal recommandé : {optimal_k} (Silhouette Score = {max(silhouette_scores):.4f})")
    return optimal_k


# =============================================================================
# PARTIE 2 : APPLICATION DU K-MEANS ET BATCHING DES COMMANDES
# =============================================================================

def apply_kmeans(data_points, k):
    """
    Applique l'algorithme K-Means sur les articles de l'entrepôt.

    :param data_points: Tableau numpy des coordonnées (x, y) de tous les articles
    :param k: Nombre de clusters
    :return: (modèle KMeans entraîné, labels des clusters pour chaque point)
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_points)
    return kmeans, labels


def batch_orders_by_cluster(orders, all_points, labels, k):
    """
    Regroupe (batching) les commandes selon leur cluster géographique dominant.

    Principe : chaque commande est assignée au cluster majoritaire de ses articles.
    Cela permet de guider la métaheuristique (ACO/AG) en regroupant les commandes
    géographiquement proches, réduisant ainsi la distance totale parcourue.

    :param orders      : Liste des commandes originales (liste de listes de (x,y))
    :param all_points  : Tableau numpy de TOUS les articles (flatten_orders de Lyna)
    :param labels      : Labels K-Means pour chaque article
    :param k           : Nombre de clusters
    :return: Un dictionnaire {cluster_id: [commandes assignées à ce cluster]}
    """
    batches = {i: [] for i in range(k)}

    # Index cumulatif pour retrouver les labels de chaque commande
    idx = 0
    for order in orders:
        n_items = len(order)
        # Labels des articles de cette commande
        order_labels = labels[idx: idx + n_items]
        # Cluster majoritaire pour cette commande
        dominant_cluster = int(np.bincount(order_labels).argmax())
        batches[dominant_cluster].append(order)
        idx += n_items

    return batches


# =============================================================================
# PARTIE 3 : VISUALISATION DES CLUSTERS
# =============================================================================

def plot_clusters(data_points, labels, centroids, depot=(0, 0), k=None):
    """
    Affiche les clusters formés par K-Means sur le plan de l'entrepôt.

    :param data_points: Coordonnées de tous les articles
    :param labels     : Labels K-Means pour chaque article
    :param centroids  : Coordonnées des centroïdes des clusters
    :param depot      : Coordonnées du dépôt (point de départ)
    :param k          : Nombre de clusters
    """
    colors = plt.cm.tab10.colors  # Palette de 10 couleurs distinctes
    plt.figure(figsize=(10, 9))

    for cluster_id in range(k):
        cluster_points = data_points[labels == cluster_id]
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            color=colors[cluster_id % len(colors)],
            s=60, label=f"Cluster {cluster_id + 1}", alpha=0.8
        )

    # Centroïdes
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        color='black', s=200, marker='X',
        label='Centroïdes', zorder=10
    )

    # Dépôt
    plt.scatter(
        depot[0], depot[1],
        color='red', s=200, marker='s',
        label='Dépôt (Départ/Arrivée)', zorder=11
    )

    plt.title(f"Clustering K-Means ({k} clusters) — Articles de l'entrepôt", fontsize=14, fontweight='bold')
    plt.xlabel("Axe X (largeur entrepôt)")
    plt.ylabel("Axe Y (longueur entrepôt)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig("clusters_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Visualisation sauvegardée : clusters_visualization.png")


# =============================================================================
# PARTIE 4 : ANALYSE DE L'IMPACT DU CLUSTERING (KPI)
# =============================================================================

def analyze_cluster_quality(data_points, labels, k):
    """
    Évalue la qualité des clusters formés et analyse leur compacité.

    Métriques calculées :
    - Silhouette Score global (cohésion + séparation)
    - Inertie par cluster (compacité interne)
    - Nombre d'articles par cluster (équilibre de la charge)
    """
    print("\n=== Analyse de la Qualité des Clusters ===")

    # Silhouette Score global
    sil_score = silhouette_score(data_points, labels)
    print(f"  Silhouette Score global : {sil_score:.4f}  (proche de 1 = excellent)")

    # Détail par cluster
    print(f"\n  Répartition des articles par cluster :")
    for i in range(k):
        count = np.sum(labels == i)
        print(f"    Cluster {i + 1} : {count} articles")

    return sil_score


def analyze_batching_impact(batches, env):
    """
    Analyse l'impact du batching sur la réduction de la complexité du problème.

    Compare :
    - Sans clustering : une seule tournée avec TOUS les articles
    - Avec clustering : K tournées indépendantes, chacune plus petite

    :param batches: Dictionnaire {cluster_id: [commandes]}
    :param env    : Instance de WarehouseEnvironment (pour le calcul des distances)
    """
    print("\n=== Impact du Clustering sur la Complexité ===")

    total_articles = sum(
        sum(len(order) for order in cluster_orders)
        for cluster_orders in batches.values()
    )

    print(f"  Nombre total d'articles    : {total_articles}")
    print(f"  Nombre de clusters         : {len(batches)}")
    print(f"  Taille moyenne d'un cluster: {total_articles / len(batches):.1f} articles")

    print(f"\n  Sans clustering → 1 tournée de {total_articles} articles")
    print(f"  Avec clustering → {len(batches)} tournées de ~{total_articles / len(batches):.0f} articles chacune")

    # Réduction de la complexité (TSP est O(n!))
    import math
    complexity_without = math.factorial(min(total_articles, 15))  # Limité pour éviter overflow
    avg_cluster_size = int(total_articles / len(batches))
    complexity_with = math.factorial(min(avg_cluster_size, 15))

    print(f"\n  Réduction de complexité (indicative, TSP est O(n!)) :")
    print(f"    Sans ML : O({min(total_articles, 15)}!) ")
    print(f"    Avec ML : O({min(avg_cluster_size, 15)}!) par cluster")
    print(f"  → Le clustering simplifie considérablement l'espace de recherche de la métaheuristique.")


# =============================================================================
# PIPELINE PRINCIPAL — Point d'entrée du script
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("   KENZA MENAD — K-Means Clustering pour Order Picking")
    print("=" * 60)

    # --- Étape 1 : Initialisation de l'entrepôt (code de Lyna) ---
    print("\n[1/5] Initialisation de l'entrepôt...")
    env = WarehouseEnvironment(width=200, height=200, depot=(0, 0))

    # Génération de commandes avec le même seed que Lyna (reproductibilité)
    orders = env.generate_orders(num_orders=30, items_per_order=(5, 10), seed=42)
    print(f"      {len(orders)} commandes générées.")

    # --- Étape 2 : Préparation des données pour K-Means ---
    print("\n[2/5] Préparation des données (flatten des commandes)...")
    all_points = env.flatten_orders(orders)
    print(f"      {len(all_points)} articles au total extraits.")

    # Visualisation initiale de l'entrepôt
    env.plot_warehouse(order_points=all_points, title="Entrepôt - Tous les articles (avant clustering)")

    # --- Étape 3 : Détermination du K optimal ---
    print("\n[3/5] Recherche du K optimal...")
    optimal_k = find_optimal_k(all_points, k_min=2, k_max=10)

    # --- Étape 4 : Application du K-Means ---
    print(f"\n[4/5] Application du K-Means avec K={optimal_k}...")
    kmeans_model, labels = apply_kmeans(all_points, k=optimal_k)
    centroids = kmeans_model.cluster_centers_

    # Visualisation des clusters
    plot_clusters(all_points, labels, centroids, depot=env.depot, k=optimal_k)

    # --- Étape 5 : Batching des commandes ---
    print("\n[5/5] Batching des commandes par cluster...")
    batches = batch_orders_by_cluster(orders, all_points, labels, k=optimal_k)

    for cluster_id, cluster_orders in batches.items():
        total_items = sum(len(o) for o in cluster_orders)
        print(f"  Cluster {cluster_id + 1} : {len(cluster_orders)} commandes | {total_items} articles")

    # --- Analyse qualité ---
    analyze_cluster_quality(all_points, labels, k=optimal_k)
    analyze_batching_impact(batches, env)

