import matplotlib.pyplot as plt
import numpy as np

# Tes données exactes
labels = ['ACO\n(Sans ML)', 'AG\n(Sans ML)', 'ACO\n(Avec ML)', 'AG\n(Avec ML)']
distances = [2588.0, 12052.0, 3334.0, 5600.0]
temps = [3.32, 5.37, 1.34, 4.42]

# ==========================================
# GRAPHIQUE 1 : Double Bar Chart (Dist/Temps)
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(labels))
width = 0.35

color1 = 'tab:blue'
ax1.set_ylabel('Distance Totale (Unités)', color=color1, fontweight='bold')
bars1 = ax1.bar(x - width/2, distances, width, label='Distance', color=color1, alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Temps de calcul (Secondes)', color=color2, fontweight='bold')
bars2 = ax2.bar(x + width/2, temps, width, label='Temps', color=color2, alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color2)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontweight='bold')
plt.title("Comparaison des Performances : Distance vs Temps", fontsize=14, fontweight='bold')

fig.tight_layout()
plt.savefig("fig_barres_performances.png", dpi=300)
print("✅ Graphique en barres généré : fig_barres_performances.png")

# ==========================================
# GRAPHIQUE 2 : Scatter Plot (Compromis Pareto)
# ==========================================
plt.figure(figsize=(8, 6))
colors = ['blue', 'orange', 'green', 'red']

for i in range(len(labels)):
    plt.scatter(temps[i], distances[i], s=200, c=colors[i], label=labels[i], edgecolors='black', zorder=5)

plt.xlabel("Temps de calcul (Secondes) → Plus bas = Mieux", fontweight='bold')
plt.ylabel("Distance Totale → Plus bas = Mieux", fontweight='bold')
plt.title("Analyse du compromis Temps / Distance", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
plt.legend(title="Approches")

# Ajout d'une zone "idéale"
plt.axvspan(0, 2, ymin=0, ymax=0.3, color='green', alpha=0.1, label="Zone Idéale")

plt.tight_layout()
plt.savefig("fig_scatter_pareto.png", dpi=300)
print("✅ Graphique nuage de points généré : fig_scatter_pareto.png")