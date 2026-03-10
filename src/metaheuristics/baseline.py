import time
# Import de la classe mère (Lyna)
from src.env.warehouse_env import WarehouseEnvironment


def nearest_neighbor_tsp(warehouse, order_points):
    """
    Algorithme Glouton (Baseline) :
    Sert de point de comparaison pour les KPIs (Distance et Temps).
    """
    # KPI : Chronomètre de précision
    start_time = time.perf_counter()

    # CORRECTION : On s'assure que order_points est une liste de tuples
    # Cela évite l'erreur d'ambiguïté de Numpy lors du .remove()
    unvisited = [tuple(p) for p in order_points]

    current_pos = tuple(warehouse.depot)
    route = [current_pos]
    total_distance = 0

    # Algorithme du plus proche voisin
    while unvisited:
        # Recherche du point le plus proche selon Manhattan
        next_pos = min(unvisited, key=lambda p: warehouse.calculate_distance(current_pos, p))

        # Accumulation de la distance (Fonction Objectif)
        total_distance += warehouse.calculate_distance(current_pos, next_pos)
        current_pos = next_pos

        route.append(current_pos)
        unvisited.remove(current_pos)  # Maintenant cela fonctionne !

    # Retour final au dépôt
    total_distance += warehouse.calculate_distance(current_pos, warehouse.depot)
    route.append(warehouse.depot)

    # KPI : Temps d'exécution final
    exec_time = time.perf_counter() - start_time

    return route, total_distance, exec_time


# --- TEST DE RÉFÉRENCE ---
if __name__ == "__main__":
    print("=" * 50)
    print("   TEST DE LA BASELINE (TSP GLOBAL SANS ML)")
    print("=" * 50)

    # 1. On initialise l'entrepôt
    env = WarehouseEnvironment(width=200, height=200)

    # 2. On génère un volume important
    raw_orders = env.generate_orders(num_orders=20, items_per_order=(8, 12), seed=42)
    tous_les_articles = env.flatten_orders(raw_orders)

    print(f"📦 Total d'articles à collecter : {len(tous_les_articles)}")

    # 3. Calcul de la tournée Baseline
    chemin, distance_totale, temps_calcul = nearest_neighbor_tsp(env, tous_les_articles)

    # 4. Affichage des KPIs pour votre rapport
    print("\n📊 KPIs DE LA SOLUTION NON-OPTIMISÉE :")
    print(f"   Distance totale parcourue : {distance_totale} m")
    print(f"   Temps de calcul           : {temps_calcul:.5f} secondes")
    print("-" * 50)

    # 5. Visualisation
    env.plot_route(
        route=chemin,
        title=f"BASELINE (Glouton Global) | Articles: {len(tous_les_articles)} | Dist: {distance_totale}"
    )