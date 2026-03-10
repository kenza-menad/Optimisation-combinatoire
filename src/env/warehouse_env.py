import numpy as np
import matplotlib.pyplot as plt


class WarehouseEnvironment:
    def __init__(self, width=100, height=100, depot=(0, 0)):
        """
        Initialise l'entrepôt.
        :param width: Largeur de l'entrepôt (axe X)
        :param height: Longueur de l'entrepôt (axe Y)
        :param depot: Coordonnées du point de départ/arrivée du préparateur
        """
        self.width = width
        self.height = height
        self.depot = depot

    def calculate_distance(self, point1, point2):
        """
        Calcule la distance de Manhattan entre deux points.
        Dans un entrepôt (avec des allées), on ne peut pas se déplacer en diagonale.
        """
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def generate_orders(self, num_orders=50, items_per_order=(5, 15)):
        """
        Génère une liste de commandes aléatoires.
        :param num_orders: Nombre total de commandes à générer.
        :param items_per_order: Tuple (min, max) du nombre d'articles par commande.
        :return: Une liste de commandes (chaque commande est une liste de coordonnées (x,y))
        """
        orders = []
        for _ in range(num_orders):
            num_items = np.random.randint(items_per_order[0], items_per_order[1] + 1)
            # Génère des coordonnées aléatoires pour chaque article de la commande
            order = [(np.random.randint(0, self.width), np.random.randint(0, self.height))
                     for _ in range(num_items)]
            orders.append(order)
        return orders

    def plot_warehouse(self, order_points=None, title="Plan de l'entrepôt"):
        """
        Affiche graphiquement l'entrepôt, le dépôt et les articles d'une commande.
        """
        plt.figure(figsize=(8, 8))

        # Affichage du dépôt (point rouge)
        plt.scatter(*self.depot, color='red', s=100, marker='s', label='Dépôt (Départ/Arrivée)')

        # Affichage des articles si une commande est fournie (points bleus)
        if order_points:
            x_coords = [p[0] for p in order_points]
            y_coords = [p[1] for p in order_points]
            plt.scatter(x_coords, y_coords, color='blue', s=50, label='Articles à collecter')

            # Ajoute un numéro à côté de chaque article pour y voir clair
            for i, (x, y) in enumerate(order_points):
                plt.text(x + 1, y + 1, str(i), fontsize=9)

        plt.title(title)
        plt.xlim(-5, self.width + 5)
        plt.ylim(-5, self.height + 5)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()


# --- TEST DU SCRIPT (À exécuter pour vérifier que tout fonctionne) ---
if __name__ == "__main__":
    # 1. Création de l'entrepôt (100x100)
    my_warehouse = WarehouseEnvironment(width=100, height=100)

    # 2. Génération de 10 commandes
    my_orders = my_warehouse.generate_orders(num_orders=10, items_per_order=(5, 10))

    # 3. On sélectionne la toute première commande pour la visualiser
    first_order = my_orders[0]
    print(f"Coordonnées des articles de la première commande : {first_order}")

    # 4. Affichage graphique
    my_warehouse.plot_warehouse(order_points=first_order, title="Visualisation d'une commande isolée")