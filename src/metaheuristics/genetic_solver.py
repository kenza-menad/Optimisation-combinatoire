import numpy as np
import random


class GeneticAlgorithmSolver:
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1):
        """
        Initialisation de l'Algorithme Génétique.
        :param population_size: Nombre de chemins différents générés à chaque génération.
        :param generations: Nombre de cycles d'évolution.
        :param mutation_rate: Probabilité (entre 0 et 1) d'inverser deux points dans un trajet.
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def solve(self, coordinates, distance_matrix):
        """
        Trouve le chemin le plus court en simulant l'évolution naturelle.
        """
        num_points = len(coordinates)
        if num_points <= 2:
            # S'il n'y a qu'un seul article, le chemin est juste [0, 1, 0]
            dist = distance_matrix[0][1] + distance_matrix[1][0] if num_points == 2 else 0
            return [0, 1, 0] if num_points == 2 else [0, 0], dist

        # 1. Création de la population initiale (des chemins aléatoires)
        # On ne mélange que les articles à visiter (de 1 à num_points - 1), le dépôt est à 0.
        articles_to_visit = list(range(1, num_points))
        population = []
        for _ in range(self.population_size):
            individual = articles_to_visit.copy()
            random.shuffle(individual)
            population.append(individual)

        best_path_global = None
        best_distance_global = float('inf')

        # 2. Boucle des générations (l'évolution)
        for generation in range(self.generations):

            # Évaluation de la population (calcul des distances)
            fitness_scores = []
            distances = []

            for ind in population:
                dist = self._calculate_distance(ind, distance_matrix)
                distances.append(dist)

                # La fitness est l'inverse de la distance : un trajet court donne un score élevé
                fitness = 1.0 / dist if dist > 0 else float('inf')
                fitness_scores.append(fitness)

                # Sauvegarde du meilleur individu absolu
                if dist < best_distance_global:
                    best_distance_global = dist
                    best_path_global = ind.copy()

            # Création de la nouvelle génération
            new_population = []

            # Élitisme : On garde toujours le meilleur chemin de la génération précédente
            best_idx = np.argmin(distances)
            new_population.append(population[best_idx])

            # Remplissage du reste de la population avec croisement et mutation
            while len(new_population) < self.population_size:
                # Sélection par "Roulette" (les meilleurs ont plus de chances d'être tirés)
                parent1 = random.choices(population, weights=fitness_scores, k=1)[0]
                parent2 = random.choices(population, weights=fitness_scores, k=1)[0]

                # Croisement
                child = self._order_crossover(parent1, parent2)

                # Mutation
                self._mutate(child)

                new_population.append(child)

            population = new_population

        # Formatage final du chemin : on rajoute le dépôt (0) au début et à la fin
        final_path = [0] + best_path_global + [0]
        return final_path, best_distance_global

    def _calculate_distance(self, individual, distance_matrix):
        """Calcule la distance totale d'un chromosome (en incluant le départ et retour au dépôt)."""
        dist = distance_matrix[0][individual[0]]  # Dépôt vers 1er article
        for i in range(len(individual) - 1):
            dist += distance_matrix[individual[i]][individual[i + 1]]
        dist += distance_matrix[individual[-1]][0]  # Dernier article vers Dépôt
        return dist

    def _order_crossover(self, parent1, parent2):
        """Croisement de type 'Order Crossover' (OX) spécifique au problème du voyageur de commerce."""
        size = len(parent1)
        child = [-1] * size

        # On choisit deux points de coupe au hasard
        start, end = sorted(random.sample(range(size), 2))

        # On copie la portion du parent 1 dans l'enfant
        child[start:end + 1] = parent1[start:end + 1]

        # On remplit les trous avec les éléments du parent 2 (en gardant leur ordre)
        p2_idx = 0
        for i in range(size):
            if child[i] == -1:
                while parent2[p2_idx] in child:
                    p2_idx += 1
                child[i] = parent2[p2_idx]

        return child

    def _mutate(self, individual):
        """Mutation par échange : on inverse la position de deux articles dans le trajet."""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]


# --- TEST LOCAL DU SCRIPT ---
if __name__ == "__main__":
    print("🧬 Test de l'Algorithme Génétique...")
    fausses_coords = [(0, 0), (2, 4), (5, 2), (7, 8), (1, 9)]
    num_points = len(fausses_coords)

    # Matrice des distances (Manhattan)
    vraie_matrice = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            x1, y1 = fausses_coords[i]
            x2, y2 = fausses_coords[j]
            vraie_matrice[i][j] = abs(x1 - x2) + abs(y1 - y2)

    optimizer = GeneticAlgorithmSolver(population_size=20, generations=50, mutation_rate=0.1)
    chemin, dist = optimizer.solve(fausses_coords, vraie_matrice)

    print("\n--- RÉSULTAT FINAL ---")
    print(f"Meilleur chemin trouvé (index) : {chemin}")
    print(f"Distance totale : {dist}")