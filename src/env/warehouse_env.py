import numpy as np
import matplotlib.pyplot as plt

class WarehouseEnvironment:
    def __init__(self, width=100, height=100, depot=(0, 0)):
        """
        Initialise l'environnement virtuel de l'entrepôt.
        :param width: Largeur de l'entrepôt (axe X)
        :param height: Longueur de l'entrepôt (axe Y)
        :param depot: Coordonnées (x, y) du point de départ/arrivée du préparateur
        """
        self.width = width
        self.height = height
        self.depot = depot

    def calculate_distance(self, point1, point2):
        """
        Calcule la distance de Manhattan entre deux points.
        Essentiel pour modéliser le déplacement dans les allées orthogonales d'un entrepôt.
        """
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def generate_orders(self, num_orders=20, items_per_order=(5, 15), seed=None):
        """
        Génère une liste de commandes (chaque commande est une liste d'articles à collecter).
        :param num_orders: Nombre total de commandes à générer.
        :param items_per_order: Tuple (min, max) définissant la taille d'une commande.
        :param seed: Permet de figer l'aléatoire pour que tout le groupe ait les mêmes données.
        :return: Une liste de listes de coordonnées [(x, y), (x, y)...]
        """
        if seed is not None:
            np.random.seed(seed)

        orders = []
        for _ in range(num_orders):
            num_items = np.random.randint(items_per_order[0], items_per_order[1] + 1)
            # Génère les coordonnées des articles
            order = [(np.random.randint(0, self.width), np.random.randint(0, self.height))
                     for _ in range(num_items)]
            orders.append(order)
        return orders

    def flatten_orders(self, orders):
        """
        Aplatit la liste de commandes en un seul grand tableau de points.
        Fonction utilitaire conçue pour préparer les données pour le Machine Learning (Clustering).
        """
        all_points = []
        for order in orders:
            all_points.extend(order)
        return np.array(all_points)

    def plot_warehouse(self, order_points=None, title="Plan de l'entrepôt"):
        """
        Affiche graphiquement l'entrepôt avec les articles sous forme de nuage de points.
        """
        plt.figure(figsize=(8, 8))

        # Affichage du dépôt (point rouge)
        plt.scatter(*self.depot, color='red', s=150, marker='s', label='Dépôt (Départ/Arrivée)', zorder=10)

        # Affichage des articles
        if order_points is not None and len(order_points) > 0:
            if isinstance(order_points, np.ndarray):
                x_coords = order_points[:, 0]
                y_coords = order_points[:, 1]
            else:
                x_coords = [p[0] for p in order_points]
                y_coords = [p[1] for p in order_points]

            plt.scatter(x_coords, y_coords, color='blue', s=50, label='Articles à collecter', zorder=5)

        plt.title(title)
        plt.xlim(-5, self.width + 5)
        plt.ylim(-5, self.height + 5)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()

    def plot_route(self, route, title="Trajet du préparateur"):
        """
        Affiche graphiquement le chemin complet emprunté par le préparateur de commandes.
        :param route: La liste ordonnée des coordonnées visitées (incluant le dépôt).
        """
        plt.figure(figsize=(8, 8))

        x_coords = [p[0] for p in route]
        y_coords = [p[1] for p in route]

        # Tracer le chemin (ligne verte)
        plt.plot(x_coords, y_coords, color='green', linestyle='-', linewidth=2, alpha=0.6, label='Trajet (TSP)')

        # Afficher les points d'arrêt (articles)
        plt.scatter(x_coords, y_coords, color='blue', s=50, zorder=5)

        # Afficher le dépôt (rouge)
        plt.scatter(*self.depot, color='red', s=150, marker='s', label='Dépôt', zorder=10)

        # Ajouter le numéro d'ordre seulement si la route n'est pas trop chargée (< 50 points)
        if len(route) < 50:
            for i in range(1, len(route) - 1):
                plt.text(route[i][0] + 1, route[i][1] + 1, str(i), fontsize=10, color='black', fontweight='bold')

        plt.title(title)
        plt.xlim(-5, self.width + 5)
        plt.ylim(-5, self.height + 5)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()


# --- TEST DU FICHIER MASTER ---
if __name__ == "__main__":
    print("Initialisation de l'entrepôt de référence...")
    # On passe à 200x200 pour avoir plus de place pour les futurs gros tests
    env = WarehouseEnvironment(width=200, height=200)

    # On teste avec 100 articles pour voir si l'environnement tient le coup
    test_orders = env.generate_orders(num_orders=10, items_per_order=(10, 10), seed=42)
    all_points = env.flatten_orders(test_orders)

    print(f"Test : {len(all_points)} articles générés.")
    env.plot_warehouse(order_points=all_points, title="Test de l'environnement avec 100 articles")