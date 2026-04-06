import numpy as np
import time
import pandas as pd
from sklearn.cluster import KMeans

from src.env.warehouse_env import WarehouseEnvironment
from src.metaheuristics.aco_solver import AntColonyOptimizer
from src.metaheuristics.genetic_solver import GeneticAlgorithmSolver


def build_distance_matrix(env, points):
    n = len(points)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = env.calculate_distance(points[i], points[j])
    return matrix


def generate_reference_data():
    """
    Données de référence cohérentes avec le rapport :
    entrepôt 200x200, seed=42, 119 articles.
    """
    env = WarehouseEnvironment(width=200, height=200, depot=(0, 0))
    orders = env.generate_orders(num_orders=16, items_per_order=(7, 8), seed=42)
    all_points = env.flatten_orders(orders)
    return env, orders, all_points


def build_kmeans_batches(points, k=4):
    """
    Clustering K-Means fixe à k=4, comme dans le rapport.
    Retourne un dictionnaire {cluster_id: array(points)}.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(points)

    batches = {}
    for cluster_id in range(k):
        batches[cluster_id] = points[labels == cluster_id]

    return batches, labels, kmeans.cluster_centers_


def run_aco(env, points, num_ants=10, num_iterations=40):
    """
    Exécute ACO sur une liste de points (sans ou avec ML selon ce qu'on lui donne).
    """
    route_points = [env.depot] + [tuple(p) for p in points]
    distance_matrix = build_distance_matrix(env, route_points)

    optimizer = AntColonyOptimizer(
        num_ants=num_ants,
        num_iterations=num_iterations
    )

    start = time.time()
    _, best_distance = optimizer.solve(route_points, distance_matrix)
    end = time.time()

    return best_distance, end - start


def run_ga(env, points, population_size=100, generations=300, mutation_rate=0.05):
    """
    Exécute l'algorithme génétique sur une liste de points.
    """
    route_points = [env.depot] + [tuple(p) for p in points]
    distance_matrix = build_distance_matrix(env, route_points)

    optimizer = GeneticAlgorithmSolver(
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate
    )

    start = time.time()
    _, best_distance = optimizer.solve(route_points, distance_matrix)
    end = time.time()

    return best_distance, end - start


def run_with_ml(env, batches, solver_name):
    """
    Applique une métaheuristique cluster par cluster, puis somme les distances.
    """
    total_distance = 0.0
    total_time = 0.0

    for cluster_id, batch_points in batches.items():
        if len(batch_points) == 0:
            continue

        if solver_name == "ACO":
            dist, exec_time = run_aco(env, batch_points, num_ants=10, num_iterations=40)
        elif solver_name == "AG":
            dist, exec_time = run_ga(env, batch_points, population_size=100, generations=300, mutation_rate=0.05)
        else:
            raise ValueError("solver_name doit être 'ACO' ou 'AG'.")

        total_distance += dist
        total_time += exec_time

    return total_distance, total_time


def main():
    print("=" * 80)
    print("COMPARAISON METAHEURISTIQUES AVEC / SANS MACHINE LEARNING")
    print("=" * 80)

    # 1) Données communes
    env, orders, all_points = generate_reference_data()
    print(f"Nombre total d'articles : {len(all_points)}")

    # 2) K-Means fixe à 4 clusters, comme dans le rapport
    batches, labels, centers = build_kmeans_batches(all_points, k=4)
    print("K-Means appliqué avec k = 4")
    for cid, pts in batches.items():
        print(f"  Cluster {cid}: {len(pts)} articles")

    # 3) ACO sans ML
    print("\n[1] ACO sans ML")
    aco_no_ml_dist, aco_no_ml_time = run_aco(env, all_points, num_ants=10, num_iterations=40)
    print(f"Distance : {aco_no_ml_dist:.1f}")
    print(f"Temps    : {aco_no_ml_time:.2f} s")

    # 4) AG sans ML
    print("\n[2] AG sans ML")
    ga_no_ml_dist, ga_no_ml_time = run_ga(env, all_points, population_size=100, generations=300, mutation_rate=0.05)
    print(f"Distance : {ga_no_ml_dist:.1f}")
    print(f"Temps    : {ga_no_ml_time:.2f} s")

    # 5) ACO avec ML
    print("\n[3] ACO avec ML (K-Means + k=4)")
    aco_ml_dist, aco_ml_time = run_with_ml(env, batches, "ACO")
    print(f"Distance : {aco_ml_dist:.1f}")
    print(f"Temps    : {aco_ml_time:.2f} s")

    # 6) AG avec ML
    print("\n[4] AG avec ML (K-Means + k=4)")
    ga_ml_dist, ga_ml_time = run_with_ml(env, batches, "AG")
    print(f"Distance : {ga_ml_dist:.1f}")
    print(f"Temps    : {ga_ml_time:.2f} s")

    # 7) Tableau final
    df = pd.DataFrame([
        ["ACO sans ML", aco_no_ml_dist, aco_no_ml_time],
        ["AG sans ML", ga_no_ml_dist, ga_no_ml_time],
        ["ACO avec ML", aco_ml_dist, aco_ml_time],
        ["AG avec ML", ga_ml_dist, ga_ml_time],
    ], columns=["Approche", "Distance totale", "Temps (s)"])

    print("\n" + "=" * 80)
    print("TABLEAU FINAL")
    print("=" * 80)
    print(df.to_string(index=False))

    df.to_csv("resultats_metaheuristiques_ml.csv", index=False)
    print("\nFichier sauvegardé : resultats_metaheuristiques_ml.csv")


if __name__ == "__main__":
    main()