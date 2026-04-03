import numpy as np
import time

# Imports de votre équipe
from src.env.warehouse_env import WarehouseEnvironment
from src.metaheuristics.aco_solver import AntColonyOptimizer

# 👇 Le pipeline de Kenza
from src.ml.compare_algorithms import run_full_pipeline


def run_hybrid_pipeline():
    print("=" * 60)
    print(" 🚀 LANCEMENT DU PIPELINE HYBRIDE : ORDER PICKING (ML + ACO)")
    print("=" * 60)

    # ---------------------------------------------------------
    # ÉTAPE 1 & 2 : ENVIRONNEMENT + MACHINE LEARNING
    # ---------------------------------------------------------
    print("\n[1/2] Génération de l'environnement & Regroupement intelligent...")
    # On laisse le script de Kenza générer les commandes et trouver le meilleur algorithme
    payload, best_algo, df = run_full_pipeline(n_orders=15, items_per_order=(3, 6), seed=42)

    batches = payload["batches"]
    depot_array = payload["depot"]

    # On récupère l'environnement de base pour la distance
    env = WarehouseEnvironment(width=100, height=100, depot=tuple(depot_array))

    total_articles = sum(len(pts) for pts in batches.values())
    print(f"      Total : {total_articles} articles à collecter.")
    print(f"      L'algorithme vainqueur est : {best_algo}. Création de {payload['n_clusters']} zones de collecte...")

    # ---------------------------------------------------------
    # ÉTAPE 3 : L'OPTIMISATION (Dyhia)
    # ---------------------------------------------------------
    print("\n[3/3] Calcul des tournées optimales avec les Fourmis (ACO)...")

    optimizer = AntColonyOptimizer(num_ants=10, num_iterations=40)
    distance_totale_globale = 0.0
    temps_debut = time.time()

    # On lance les fourmis sur chaque "Lot" (Cluster)
    for cluster_id, batch_pts in batches.items():
        if len(batch_pts) == 0: continue

        # On ajoute le dépôt (0,0) au tout début du trajet !
        points_a_visiter = [env.depot] + [tuple(p) for p in batch_pts]
        num_points = len(points_a_visiter)

        # Calcul de la matrice des distances pour CE cluster
        matrice_distance = np.zeros((num_points, num_points))
        for j in range(num_points):
            for k in range(num_points):
                matrice_distance[j][k] = env.calculate_distance(points_a_visiter[j], points_a_visiter[k])

        # 🐜 Les fourmis entrent en action !
        print(f"      -> Résolution du Cluster {cluster_id + 1} ({num_points - 1} articles)... ", end="")
        chemin, dist = optimizer.solve(points_a_visiter, matrice_distance)
        print(f"Terminé ! Distance = {dist:.1f}")

        distance_totale_globale += dist

    temps_fin = time.time()

    # ---------------------------------------------------------
    # RÉSULTATS FINAUX
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" 🏆 RÉSULTATS DE LA SIMULATION")
    print("=" * 60)
    print(f" Distance totale parcourue : {distance_totale_globale:.1f} unités")
    print(f" Temps de calcul de l'ACO  : {temps_fin - temps_debut:.2f} secondes")
    print("=" * 60)


if __name__ == "__main__":
    run_hybrid_pipeline()