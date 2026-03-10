from src.env.warehouse_env import WarehouseEnvironment
from src.metaheuristics.baseline import nearest_neighbor_tsp


def run_baseline_experiments():
    print("🚀 Lancement des expérimentations de la Baseline (Glouton)\n")
    print(f"{'Taille Commande':<18} | {'Distance Totale':<18} | {'Temps (secondes)':<18}")
    print("-" * 60)

    env = WarehouseEnvironment(width=100, height=100)
    tailles_test = [10, 30, 50, 100]  # On augmente la difficulté !

    for taille in tailles_test:
        # On génère une commande de la taille demandée
        commandes = env.generate_orders(num_orders=1, items_per_order=(taille, taille), seed=42)

        # On calcule les KPIs
        _, distance, temps = nearest_neighbor_tsp(env, commandes[0])

        # On affiche la ligne du tableau
        print(f"{taille:<18} | {distance:<18} | {temps:.5f}")


if __name__ == "__main__":
    run_baseline_experiments()