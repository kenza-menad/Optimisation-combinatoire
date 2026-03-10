import numpy as np
import time

# Imports de votre équipe
from src.env.warehouse_env import WarehouseEnvironment
from src.kmeans_clustering import apply_kmeans, batch_orders_by_cluster, find_optimal_k
from src.metaheuristics.aco_solver import AntColonyOptimizer


def run_hybrid_pipeline():
    print("=" * 60)
    print(" 🚀 LANCEMENT DU PIPELINE HYBRIDE : ORDER PICKING (ML + ACO)")
    print("=" * 60)

    # ---------------------------------------------------------
    # ÉTAPE 1 : L'ENVIRONNEMENT (Lyna)
    # ---------------------------------------------------------
    print("\n[1/3] Génération de l'environnement et des commandes...")
    env = WarehouseEnvironment(width=100, height=100, depot=(0, 0))

    # On génère 15 commandes contenant chacune entre 3 et 6 articles
    orders = env.generate_orders(num_orders=15, items_per_order=(3, 6))

    # Petite sécurité : si Lyna n'a pas encore codé flatten_orders, on le fait ici
    all_points = []
    for order in orders:
        all_points.extend(order)
    all_points = np.array(all_points)

    print(f"      Total : {len(orders)} commandes générées, représentant {len(all_points)} articles à collecter.")

    # ---------------------------------------------------------
    # ÉTAPE 2 : LE MACHINE LEARNING (Kenza)
    # ---------------------------------------------------------
    print("\n[2/3] Regroupement intelligent avec K-Means...")
    # On fixe un K=4 pour le test (ou tu peux utiliser sa fonction find_optimal_k)
    k_optimal = 4
    print(f"      Application du K-Means pour créer {k_optimal} zones de collecte (clusters)...")

    kmeans_model, labels = apply_kmeans(all_points, k=k_optimal)

    # On utilise la super fonction de Kenza pour regrouper les commandes
    batches = batch_orders_by_cluster(orders, all_points, labels, k=k_optimal)

    # ---------------------------------------------------------
    # ÉTAPE 3 : L'OPTIMISATION (Dyhia)
    # ---------------------------------------------------------
    print("\n[3/3] Calcul des tournées optimales avec les Fourmis (ACO)...")

    optimizer = AntColonyOptimizer(num_ants=10, num_iterations=40)
    distance_totale_globale = 0.0
    temps_debut = time.time()

    # On lance tes fourmis sur chaque "Lot" (Cluster) préparé par Kenza
    for cluster_id, cluster_orders in batches.items():
        if not cluster_orders:
            continue  # Si un cluster est vide, on passe au suivant

        # On extrait tous les articles de ce lot
        articles_du_lot = []
        for order in cluster_orders:
            articles_du_lot.extend(order)

        # On ajoute le dépôt (0,0) au tout début du trajet !
        points_a_visiter = [env.depot] + articles_du_lot
        num_points = len(points_a_visiter)

        # Calcul de la matrice des distances pour CE cluster
        matrice_distance = np.zeros((num_points, num_points))
        for j in range(num_points):
            for k in range(num_points):
                matrice_distance[j][k] = env.calculate_distance(points_a_visiter[j], points_a_visiter[k])

        # 🐜 Tes fourmis entrent en action !
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