import numpy as np
import random

class AntColonyOptimizer:
    def __init__(self, num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5):
        """
        Initialisation de l'algorithme des Colonies de Fourmis (ACO).
        :param num_ants: Nombre de fourmis (agents)
        :param num_iterations: Nombre de cycles de recherche
        :param alpha: Importance des phéromones (la trace laissée)
        :param beta: Importance de la visibilité (la distance réelle)
        :param evaporation_rate: Vitesse à laquelle les phéromones disparaissent
        """
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate

    def solve(self, coordinates, distance_matrix):
        """
        Trouve le chemin le plus court pour relier tous les points.
        :param coordinates: Liste des coordonnées (x,y) des articles à visiter
        :param distance_matrix: Matrice des distances entre chaque point
        :return: Le meilleur chemin (liste d'index) et la distance totale
        """
        num_points = len(coordinates)
        # 1. Initialiser la matrice des phéromones (au début, il y en a partout pareil)
        pheromones = np.ones((num_points, num_points))

        best_path = None
        best_distance = float('inf')

        # Boucle principale de l'algorithme
        for iteration in range(self.num_iterations):
            all_paths = []
            all_distances = []

            # ÉTAPE A : Chaque fourmi construit une solution
            for ant in range(self.num_ants):
                path, distance = self._construct_path(pheromones, distance_matrix)
                all_paths.append(path)
                all_distances.append(distance)

                # Sauvegarder si c'est le meilleur chemin trouvé jusqu'à présent
                if distance < best_distance:
                    best_distance = distance
                    best_path = path

            # ÉTAPE B : Mise à jour des phéromones (Évaporation + Dépôt des nouvelles)
            self._update_pheromones(pheromones, all_paths, all_distances)

            # Optionnel : Afficher la progression de temps en temps
            if iteration % 10 == 0:
                print(f"Itération {iteration}: Meilleure distance = {best_distance}")

        return best_path, best_distance

    def _construct_path(self, pheromones, distance_matrix):
        num_points = len(distance_matrix)

        # En logistique, le préparateur commence toujours au dépôt (index 0)
        current_node = 0
        path = [current_node]
        total_distance = 0.0

        # Liste des articles qu'il reste à visiter (tous sauf le dépôt)
        unvisited = list(range(1, num_points))

        while unvisited:
            probabilities = []

            # On calcule l'attractivité de chaque point non visité
            for next_node in unvisited:
                # 1. Attractivité liée à la phéromone (historique)
                tau = pheromones[current_node][next_node]

                # 2. Attractivité liée à la distance (glouton)
                # On met un tout petit chiffre (1e-6) pour éviter la division par zéro si la distance est 0
                dist = distance_matrix[current_node][next_node]
                eta = 1.0 / dist if dist > 0 else 1e-6

                # Calcul du score brut (numérateur de la formule)
                score = (tau ** self.alpha) * (eta ** self.beta)
                probabilities.append(score)

            # Normalisation : on transforme les scores en pourcentages/probabilités (pour que la somme fasse 1)
            sum_probs = sum(probabilities)
            if sum_probs == 0:
                # Sécurité : si tout est à 0, on donne une chance égale à chaque point
                probabilities = [1.0 / len(unvisited)] * len(unvisited)
            else:
                probabilities = [p / sum_probs for p in probabilities]

            # La fourmi "jette les dés" pour choisir le prochain point selon les probabilités
            # La fonction np.random.choice est parfaite pour simuler cette roulette
            chosen_index = np.random.choice(len(unvisited), p=probabilities)
            next_node = unvisited[chosen_index]

            # On met à jour le trajet
            path.append(next_node)
            total_distance += distance_matrix[current_node][next_node]

            # On se déplace (le point devient le point actuel et est retiré des "non visités")
            current_node = next_node
            unvisited.pop(chosen_index)

        # Fin de la commande : le préparateur doit retourner au dépôt pour déposer le chariot
        total_distance += distance_matrix[current_node][path[0]]
        path.append(path[0])

        return path, total_distance

    def _update_pheromones(self, pheromones, all_paths, all_distances):
        """
        Met à jour la matrice des phéromones après qu'un groupe de fourmis a terminé son trajet.
        """
        # 1. Évaporation : on réduit toutes les pistes existantes
        # Cela permet d'oublier les mauvais chemins au fil du temps
        pheromones *= (1.0 - self.evaporation_rate)

        # 2. Dépôt : les fourmis de cette itération ajoutent de la phéromone
        for path, distance in zip(all_paths, all_distances):

            # Plus le chemin d'une fourmi est court, plus elle dépose de phéromone (Q / distance)
            # On utilise 1.0 comme constante Q
            pheromone_to_add = 1.0 / distance if distance > 0 else 0.0

            # On parcourt le trajet de la fourmi (point par point)
            for i in range(len(path) - 1):
                node_a = path[i]
                node_b = path[i + 1]

                # On ajoute la phéromone sur l'arête (dans les deux sens, car le chemin A->B est le même que B->A)
                pheromones[node_a][node_b] += pheromone_to_add
                pheromones[node_b][node_a] += pheromone_to_add
# --- TEST DE TA PARTIE (Sans attendre Lyna ou Kenza) ---
if __name__ == "__main__":
    print("Test de l'ACO de Dyhia...")
    # On simule 5 points (articles) pour tester
    fausses_coords = [(0, 0), (2, 4), (5, 2), (7, 8), (1, 9)]
    # On crée une fausse matrice de distance remplie de 1 (juste pour que le code tourne)
    fausse_matrice = np.ones((5, 5))

    optimizer = AntColonyOptimizer(num_ants=5, num_iterations=50)
    chemin, dist = optimizer.solve(fausses_coords, fausse_matrice)
    print("Fin du test de l'algorithme.")

    # --- TEST DE TA PARTIE (Avec de vraies distances) ---
    if __name__ == "__main__":
        print("Test de l'ACO de Dyhia...")
        # On simule 5 points (articles) pour tester
        fausses_coords = [(0, 0), (2, 4), (5, 2), (7, 8), (1, 9)]
        num_points = len(fausses_coords)

        # On calcule la VRAIE matrice des distances (Distance de Manhattan)
        vraie_matrice = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                x1, y1 = fausses_coords[i]
                x2, y2 = fausses_coords[j]
                vraie_matrice[i][j] = abs(x1 - x2) + abs(y1 - y2)

        # On lance l'optimiseur !
        optimizer = AntColonyOptimizer(num_ants=10, num_iterations=50)
        chemin, dist = optimizer.solve(fausses_coords, vraie_matrice)

        print("\n--- RÉSULTAT FINAL ---")
        print(f"Meilleur chemin trouvé (index) : {chemin}")
        print(f"Distance totale : {dist}")