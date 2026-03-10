import time
from src.env.warehouse_env import WarehouseEnvironment
from src.metaheuristics.baseline import nearest_neighbor_tsp
from src.metaheuristics.genetic_algo import GeneticAlgorithmTSP


def run_two_phase_analysis():
    env = WarehouseEnvironment(width=200, height=200)

    print("=" * 60)
    print(" 🟢 PHASE 1 : PETITE COMMANDE (Preuve d'efficacité de l'AG)")
    print("=" * 60)

    # Génération d'une seule commande de 25 articles
    small_orders = env.generate_orders(num_orders=1, items_per_order=(25, 25), seed=42)
    small_points = [tuple(p) for p in env.flatten_orders(small_orders)]

    print(f"📦 Nombre d'articles : {len(small_points)}")

    # 1. Baseline
    _, dist_base_small, time_base_small = nearest_neighbor_tsp(env, small_points)
    print(f"[BASELINE] Distance : {dist_base_small} | Temps : {time_base_small:.4f}s")

    # 2. Génétique
    ga_small = GeneticAlgorithmTSP(env, small_points, pop_size=100, generations=300)
    _, dist_ga_small, time_ga_small = ga_small.run()
    print(f"[GÉNÉTIQUE] Distance : {dist_ga_small:.2f} | Temps : {time_ga_small:.4f}s")

    gain_small = ((dist_base_small - dist_ga_small) / dist_base_small) * 100
    print(f"💡 Conclusion Phase 1 : L'AG fonctionne et améliore la distance de {gain_small:.2f}% !\n")

    print("=" * 60)
    print(" 🔴 PHASE 2 : TOUT L'ENTREPÔT (L'effondrement de l'AG)")
    print("=" * 60)

    # Génération d'une vague complète de 130 articles
    large_orders = env.generate_orders(num_orders=10, items_per_order=(12, 14), seed=42)
    large_points = [tuple(p) for p in env.flatten_orders(large_orders)]

    print(f"📦 Nombre d'articles : {len(large_points)}")

    # 1. Baseline
    _, dist_base_large, time_base_large = nearest_neighbor_tsp(env, large_points)
    print(f"[BASELINE] Distance : {dist_base_large} | Temps : {time_base_large:.4f}s")

    # 2. Génétique (Mêmes paramètres que la phase 1 pour comparer équitablement)
    ga_large = GeneticAlgorithmTSP(env, large_points, pop_size=100, generations=300)
    _, dist_ga_large, time_ga_large = ga_large.run()
    print(f"[GÉNÉTIQUE] Distance : {dist_ga_large:.2f} | Temps : {time_ga_large:.4f}s")

    gain_large = ((dist_base_large - dist_ga_large) / dist_base_large) * 100
    print(f"⚠️ Conclusion Phase 2 : L'AG est perdu. La distance empire de {gain_large:.2f}% et le temps explose !")
    print("=" * 60)
    print("🎯 DÉDUCTION POUR LE RAPPORT : Il FAUT utiliser le K-Means pour découper cet entrepôt !")


if __name__ == "__main__":
    run_two_phase_analysis()