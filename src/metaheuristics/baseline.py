# On importe la classe que tu as codée précédemment !
from src.env.warehouse_env import WarehouseEnvironment


def nearest_neighbor_routing(warehouse, order):
    """
    Calcule une tournée en utilisant l'algorithme Glouton (Plus Proche Voisin).
    :param warehouse: L'objet WarehouseEnvironment (pour utiliser sa fonction de distance)
    :param order: Une liste de coordonnées (x,y) représentant les articles à collecter
    :return: Un tuple (route_optimisée, distance_totale)
    """
    unvisited = order.copy()  # Copie la liste pour ne pas effacer l'originale
    current_location = warehouse.depot
    route = [current_location]
    total_distance = 0

    # Tant qu'il reste des articles à aller chercher
    while unvisited:
        # Trouver l'article le plus proche de notre position actuelle
        next_point = min(unvisited, key=lambda point: warehouse.calculate_distance(current_location, point))

        # Ajouter la distance parcourue
        total_distance += warehouse.calculate_distance(current_location, next_point)

        # Se déplacer vers ce nouveau point
        current_location = next_point
        route.append(current_location)
        unvisited.remove(current_location)

    # Une fois le chariot plein, il faut retourner au dépôt !
    total_distance += warehouse.calculate_distance(current_location, warehouse.depot)
    route.append(warehouse.depot)

    return route, total_distance


# --- TEST DU SCRIPT ---
if __name__ == "__main__":
    # 1. On crée un petit entrepôt et une commande
    env = WarehouseEnvironment(width=50, height=50)
    ma_commande = env.generate_orders(num_orders=1, items_per_order=(8, 8))[0]

    print(f"Articles à récupérer : {ma_commande}")

    # 2. On lance notre algorithme glouton !
    chemin_calculé, distance_finale = nearest_neighbor_routing(env, ma_commande)

    print(f"\nChemin emprunté par le préparateur : {chemin_calculé}")
    print(f"Distance totale parcourue : {distance_finale} unités")