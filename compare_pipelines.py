import numpy as np
import time

from src.env.warehouse_env import WarehouseEnvironment
from src.metaheuristics.aco_solver import AntColonyOptimizer
from src.metaheuristics.genetic_solver import GeneticAlgorithmSolver
from src.ml.compare_algorithms import run_full_pipeline


def run_ultimate_comparison():
    print("=" * 70)
    print(" ⚔️ LE GRAND AFFRONTEMENT : ACO vs ALGORYTHME GÉNÉTIQUE")
    print("=" * 70)

    # --- 1. GÉNÉRATION DES DONNÉES & MACHINE LEARNING ---
    print("\n🧠 Lancement du pipeline dynamique de Clustering (Kenza)...")
    # Au lieu de générer l'environnement à la main, on laisse le pipeline s'en charger
    # Il va tester tous les algos (K-Means, DBSCAN...) et garder le meilleur !
    payload, best_algo, df = run_full_pipeline(n_orders=15, items_per_order=(3, 6), seed=42)

    batches = payload["batches"]
    depot_array = payload["depot"]

    # On recrée un environnement juste pour utiliser sa méthode calculate_distance
    env = WarehouseEnvironment(width=100, height=100, depot=tuple(depot_array))

    # On rassemble tous les points générés pour l'Approche A (sans ML)
    all_points = np.vstack(list(batches.values()))
    print(f"\n📦 Total : {len(all_points)} articles à collecter répartis par {best_algo}.\n")

    # =========================================================
    # APPROCHE A : ACO CLASSIQUE (Sans ML)
    # =========================================================
    print("=" * 70)
    print(" [APPROCHE A] ACO Standard (1 seule grande tournée, sans ML)")
    print("=" * 70)
    points_a_visiter_complet = [env.depot] + [tuple(p) for p in all_points]
    num_points_total = len(points_a_visiter_complet)

    matrice_distance_complete = np.zeros((num_points_total, num_points_total))
    for j in range(num_points_total):
        for k in range(num_points_total):
            matrice_distance_complete[j][k] = env.calculate_distance(points_a_visiter_complet[j],
                                                                     points_a_visiter_complet[k])

    optimizer_standard = AntColonyOptimizer(num_ants=10, num_iterations=40)

    debut_A = time.time()
    _, dist_A = optimizer_standard.solve(points_a_visiter_complet, matrice_distance_complete)
    fin_A = time.time()

    print(f"✅ Distance totale : {dist_A:.1f} unités")
    print(f"⏱️ Temps de calcul : {fin_A - debut_A:.2f} secondes\n")

    # =========================================================
    # APPROCHE B : ACO HYBRIDE (Fourmis + ML)
    # =========================================================
    print("=" * 70)
    print(f" [APPROCHE B] ACO Hybride (Fourmis sur clusters {best_algo})")
    print("=" * 70)
    optimizer_aco = AntColonyOptimizer(num_ants=10, num_iterations=40)
    dist_B = 0.0
    debut_B = time.time()

    for cluster_id, batch_pts in batches.items():
        if len(batch_pts) == 0: continue

        # Beaucoup plus simple qu'avant : on prend juste les points du cluster !
        points_cluster = [env.depot] + [tuple(p) for p in batch_pts]
        num_points = len(points_cluster)

        matrice_cluster = np.zeros((num_points, num_points))
        for j in range(num_points):
            for k in range(num_points):
                matrice_cluster[j][k] = env.calculate_distance(points_cluster[j], points_cluster[k])

        _, dist_cluster = optimizer_aco.solve(points_cluster, matrice_cluster)
        dist_B += dist_cluster

    fin_B = time.time()
    print(f"✅ Distance totale : {dist_B:.1f} unités")
    print(f"⏱️ Temps de calcul : {fin_B - debut_B:.2f} secondes\n")

    # =========================================================
    # APPROCHE C : AG HYBRIDE (Génétique + ML)
    # =========================================================
    print("=" * 70)
    print(f" [APPROCHE C] Algorithme Génétique Hybride (ADN sur clusters {best_algo})")
    print("=" * 70)
    optimizer_ag = GeneticAlgorithmSolver(population_size=20, generations=50, mutation_rate=0.1)
    dist_C = 0.0
    debut_C = time.time()

    for cluster_id, batch_pts in batches.items():
        if len(batch_pts) == 0: continue

        points_cluster = [env.depot] + [tuple(p) for p in batch_pts]
        num_points = len(points_cluster)

        matrice_cluster = np.zeros((num_points, num_points))
        for j in range(num_points):
            for k in range(num_points):
                matrice_cluster[j][k] = env.calculate_distance(points_cluster[j], points_cluster[k])

        _, dist_cluster = optimizer_ag.solve(points_cluster, matrice_cluster)
        dist_C += dist_cluster

    fin_C = time.time()
    print(f"✅ Distance totale : {dist_C:.1f} unités")
    print(f"⏱️ Temps de calcul : {fin_C - debut_C:.2f} secondes\n")


if __name__ == "__main__":
    run_ultimate_comparison()