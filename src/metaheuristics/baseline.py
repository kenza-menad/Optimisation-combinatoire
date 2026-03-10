import time
# On importe l'environnement créé par Lyna
from src.env.warehouse_env import WarehouseEnvironment


def nearest_neighbor_tsp(warehouse, order_points):
    """
    Algorithme de référence (Baseline) pour le TSP : Le Plus Proche Voisin (Glouton).
    Il permet d'établir la ligne de base pour évaluer la Fonction Objectif et les KPIs.

    :param warehouse: Instance de WarehouseEnvironment (pour la fonction de distance)
    :param order_points: Liste des coordonnées des articles à collecter
    :return: route (liste des points), total_distance (int), exec_time (float)
    """
    # KPI : Démarrage du chronomètre pour le temps de calcul
    start_time = time.perf_counter()

    unvisited = list(order_points)  # Copie de la liste des points à visiter
    current_pos = warehouse.depot  # Le préparateur commence au dépôt
    route = [current_pos]  # Historique du trajet
    total_distance = 0  # Fonction Objectif à minimiser

    # Tant qu'il reste des articles à collecter dans la commande
    while unvisited:
        # On cherche l'article le plus proche de notre position actuelle (Distance de Manhattan)
        next_pos = min(unvisited, key=lambda p: warehouse.calculate_distance(current_pos, p))

        # On met à jour la distance totale et la position
        total_distance += warehouse.calculate_distance(current_pos, next_pos)
        current_pos = next_pos

        # On ajoute le point au trajet et on le retire de la liste des restants
        route.append(current_pos)
        unvisited.remove(current_pos)

    # Retour au dépôt obligatoire à la fin de la tournée
    total_distance += warehouse.calculate_distance(current_pos, warehouse.depot)
    route.append(warehouse.depot)

    # KPI : Arrêt du chronomètre
    exec_time = time.perf_counter() - start_time

    return route, total_distance, exec_time


# --- TEST ET EXTRACTION DES KPIs ---
if __name__ == "__main__":
    print("--- LANCEMENT DE LA BASELINE (TSP SANS ML) ---")

    # 1. Initialisation de l'environnement (Travail de Lyna)
    env = WarehouseEnvironment(width=100, height=100)

    # 2. Génération d'une grosse commande de test (seed=42 pour la stabilité des résultats)
    test_orders = env.generate_orders(num_orders=1, items_per_order=(25, 25), seed=42)
    commande_a_traiter = test_orders[0]

    print(f"Nombre d'articles à collecter : {len(commande_a_traiter)}")

    # 3. Exécution de l'algorithme de référence
    chemin, distance_totale, temps_calcul = nearest_neighbor_tsp(env, commande_a_traiter)

    # 4. Affichage des KPIs pour le rapport
    print("\n📊 RÉSULTATS DES KPIs (Ligne de base) :")
    print(f"-> Qualité de la solution (Distance totale) : {distance_totale} unités")
    print(f"-> Temps de calcul : {temps_calcul:.5f} secondes")

    # 5. Visualisation graphique du TSP non-optimisé
    env.plot_route(
        route=chemin,
        title=f"Baseline TSP (Glouton) | Distance: {distance_totale} | Temps: {temps_calcul:.4f}s"
    )