import numpy as np
import matplotlib.pyplot as plt
from src.env.warehouse_env import WarehouseEnvironment
from src.metaheuristics.baseline import nearest_neighbor_tsp
from src.metaheuristics.genetic_algo import GeneticAlgorithmTSP


def generate_distance_curve():
    print("=" * 60)
    print(" 📊 LANCEMENT DE L'ANALYSE DE MISE À L'ÉCHELLE (SCALING)")
    print("=" * 60)

    # On garde l'environnement classique (Densité moyenne)
    env = WarehouseEnvironment(width=200, height=200)

    # Les tailles de commandes qu'on va tester
    tailles_articles = [10, 20, 30, 40, 60, 80, 100]

    distances_baseline = []
    distances_ag = []

    for taille in tailles_articles:
        print(f"🔄 Test avec {taille} articles en cours...")

        # On génère une commande de la taille spécifique
        orders = env.generate_orders(num_orders=1, items_per_order=(taille, taille), seed=42)
        points = [tuple(p) for p in env.flatten_orders(orders)]

        # 1. Calcul Baseline
        _, dist_base, _ = nearest_neighbor_tsp(env, points)
        distances_baseline.append(dist_base)

        # 2. Calcul Algorithme Génétique
        # (Paramètres constants pour montrer sa limite)
        ga = GeneticAlgorithmTSP(env, points, pop_size=200, generations=1000)
        _, dist_ag, _, _ = ga.run()
        distances_ag.append(dist_ag)

        gain = ((dist_base - dist_ag) / dist_base) * 100
        signe = "+" if gain > 0 else ""
        print(f"   ➜ Baseline: {dist_base} | AG: {dist_ag:.0f} | Gain: {signe}{gain:.1f}%")

    # --- CRÉATION DU GRAPHIQUE ---
    plt.figure(figsize=(10, 6))

    # Tracé des lignes
    plt.plot(tailles_articles, distances_baseline, marker='o', color='royalblue', label='Baseline (Glouton)',
             linewidth=2.5, markersize=8)
    plt.plot(tailles_articles, distances_ag, marker='s', color='crimson', label='Algorithme Génétique (AG)',
             linewidth=2.5, markersize=8)

    # Remplissage des zones de victoire/défaite pour bien comprendre l'analyse
    plt.fill_between(
        tailles_articles, distances_baseline, distances_ag,
        where=(np.array(distances_ag) < np.array(distances_baseline)),
        color='green', alpha=0.1, interpolate=True, label="Zone de victoire de l'AG (Gain)"
    )
    plt.fill_between(
        tailles_articles, distances_baseline, distances_ag,
        where=(np.array(distances_ag) >= np.array(distances_baseline)),
        color='red', alpha=0.1, interpolate=True, label="Zone d'effondrement de l'AG (Perte)"
    )

    plt.title("Évolution des Performances : Baseline vs Algorithme Génétique", fontsize=14, fontweight='bold')
    plt.xlabel("Nombre d'articles à collecter", fontsize=12)
    plt.ylabel("Distance Totale (Manhattan)", fontsize=12)

    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Affichage du graphique
    plt.show()


if __name__ == "__main__":
    generate_distance_curve()