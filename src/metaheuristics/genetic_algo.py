import numpy as np
import random
import time
from src.env.warehouse_env import WarehouseEnvironment
from src.metaheuristics.baseline import nearest_neighbor_tsp


class GeneticAlgorithmTSP:
    def __init__(self, warehouse, points, pop_size=100, generations=300, mutation_rate=0.05):
        self.warehouse = warehouse
        self.points = points  # Liste de tuples (x, y)
        self.num_points = len(points)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def _calculate_distance(self, route):
        """ Calcule la distance totale d'un trajet (Fonction Objectif) """
        dist = 0
        current = self.warehouse.depot
        for idx in route:
            dist += self.warehouse.calculate_distance(current, self.points[idx])
            current = self.points[idx]
        dist += self.warehouse.calculate_distance(current, self.warehouse.depot)
        return dist

    def _create_individual(self):
        """ Crée un trajet aléatoire (une permutation des indices) """
        route = list(range(self.num_points))
        random.shuffle(route)
        return route

    def _selection(self, population, fitnesses):
        """ Sélection par Tournoi : on prend 3 individus au hasard et on garde le meilleur """
        tournament_size = 3
        selected_indices = random.sample(range(self.pop_size), tournament_size)
        best_idx = max(selected_indices, key=lambda i: fitnesses[i])
        return population[best_idx]

    def _order_crossover(self, parent1, parent2):
        """
        Order Crossover (OX) : Essentiel pour le TSP.
        Copie une sous-partie du parent 1, et remplit le reste avec le parent 2 dans l'ordre,
        sans jamais dupliquer de ville.
        """
        start, end = sorted(random.sample(range(self.num_points), 2))
        child = [-1] * self.num_points

        # 1. Copier la sous-partie du parent 1
        child[start:end] = parent1[start:end]

        # 2. Remplir avec les gènes du parent 2 qui ne sont pas encore dans l'enfant
        p2_filtered = [gene for gene in parent2 if gene not in child]

        idx_p2 = 0
        for i in range(self.num_points):
            if child[i] == -1:
                child[i] = p2_filtered[idx_p2]
                idx_p2 += 1

        return child

    def _mutate(self, route):
        """ Swap Mutation : Échange la position de deux articles dans le trajet """
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(self.num_points), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        return route

    def run(self):
        """ Boucle principale de l'Algorithme Génétique """
        start_time = time.perf_counter()

        # Initialisation de la population
        population = [self._create_individual() for _ in range(self.pop_size)]

        best_route_global = None
        best_dist_global = float('inf')

        # 🟢 NOUVEAU : Création de la liste d'historique
        history = []

        for gen in range(self.generations):
            # Évaluation (Fitness = 1 / distance, on veut maximiser la fitness)
            distances = [self._calculate_distance(ind) for ind in population]
            fitnesses = [1.0 / d for d in distances]

            # Sauvegarde du meilleur individu de la génération
            min_dist_gen = min(distances)
            if min_dist_gen < best_dist_global:
                best_dist_global = min_dist_gen
                best_route_global = population[distances.index(min_dist_gen)]

            # 🟢 NOUVEAU : On enregistre la meilleure distance connue à cette génération
            history.append(best_dist_global)

            # Création de la nouvelle génération
            new_population = []
            # Élitisme : On garde le meilleur individu tel quel pour ne pas régresser
            new_population.append(best_route_global)

            while len(new_population) < self.pop_size:
                # Sélection
                p1 = self._selection(population, fitnesses)
                p2 = self._selection(population, fitnesses)
                # Croisement
                child = self._order_crossover(p1, p2)
                # Mutation
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

        exec_time = time.perf_counter() - start_time

        # Convertir les indices en coordonnées (x,y) pour l'affichage
        final_route_coords = [self.points[idx] for idx in best_route_global]

        # 🟢 NOUVEAU : On retourne l'historique en 4ème position !
        return final_route_coords, best_dist_global, exec_time, history
# --- TEST ET COMPARAISON SUR UN GROS VOLUME ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt # 🟢 NOUVEAU : pour tracer la courbe

    print("🧬 Lancement de l'Algorithme Génétique...")
    env = WarehouseEnvironment(width=200, height=200)

    # 1. On génère 10 commandes de 10 à 15 articles
    raw_orders = env.generate_orders(num_orders=10, items_per_order=(10, 15), seed=42)

    # 2. On aplatit TOUT pour faire une tournée géante
    points_bruts = env.flatten_orders(raw_orders)
    points_tuples = [tuple(p) for p in points_bruts]  # Format sûr

    print(f"📦 Total d'articles à optimiser : {len(points_tuples)}")

    # 3. Test de la Baseline (Glouton)
    route_base, dist_base, t_base = nearest_neighbor_tsp(env, points_tuples)
    print(f"\n[BASELINE] Distance : {dist_base} | Temps : {t_base:.4f}s")

    # 4. Test de l'Algorithme Génétique
    # ⚠️ ATTENTION : Avec ~125 articles, l'AG va prendre beaucoup plus de temps (peut-être 10 à 30 secondes !)
    ga = GeneticAlgorithmTSP(env, points_tuples, pop_size=100, generations=300)
    route_ga, dist_ga, t_ga, history = ga.run() # 🟢 MODIFIÉ : on ajoute history ici
    print(f"[GÉNÉTIQUE] Distance : {dist_ga:.2f} | Temps : {t_ga:.4f}s")

    # Calcul du gain
    amelioration = ((dist_base - dist_ga) / dist_base) * 100
    print(f"\n💡 L'Algorithme Génétique a réduit la distance de {amelioration:.2f}% !")

    # 🟢 NOUVEAU : Affichage de la Courbe de Convergence
    plt.figure(figsize=(8, 5))
    plt.plot(history, color='purple', linewidth=2)
    # Ligne rouge de référence
    plt.axhline(y=dist_base, color='red', linestyle='--', label='Score Glouton (Baseline)')
    plt.title("Courbe de convergence de l'Algorithme Génétique")
    plt.xlabel("Générations")
    plt.ylabel("Distance Totale (Plus bas = Meilleur)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Visualisation comparative
    env.plot_route(route_base, title=f"Baseline (Toutes commandes) - Dist: {dist_base}")
    env.plot_route(route_ga, title=f"Génétique (Toutes commandes) - Dist: {dist_ga:.2f}")