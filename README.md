#  Optimisation des tournées en entrepôt : Hybridation du problème du Voyageur de Commerce (TSP) et du Clustering

**Projet de Master 1 MIAGE - Optimisation Combinatoire**

Ce projet propose une solution logicielle innovante pour optimiser la préparation de commandes (*Order Picking*) dans un entrepôt. Face à la complexité NP-Difficile du problème du voyageur de commerce (TSP) sur de larges volumes, notre approche combine des techniques d'Apprentissage Automatique (Clustering) et des Métaheuristiques (Colonies de Fourmis, Algorithme Génétique) selon un principe de "Diviser pour Régner".

---

##  Fonctionnalités Principales

* **Environnement Virtuel Modulaire :** Modélisation d'un entrepôt 2D avec génération aléatoire de commandes et calcul des distances de Manhattan.
* **Pipeline ML Dynamique :** Au lieu d'imposer un seul modèle, le système évalue dynamiquement plusieurs algorithmes de clustering (K-Means, DBSCAN, Gaussian Mixture, Agglomerative) et sélectionne le meilleur selon le score de Silhouette.
* **Métaheuristiques de Routage :** Résolution du TSP intra-cluster via l'Algorithme des Colonies de Fourmis (ACO) et un Algorithme Génétique (AG) adaptés.
* **Génération Automatique de Rapports :** Création de visualisations graphiques et de tableaux de métriques pour l'analyse des performances.

---

##  Architecture du Projet

L'architecture respecte le principe de séparation des préoccupations pour garantir un code maintenable et évolutif :

```text
 Optimisation-combinatoire
 ┣  src
 ┃ ┣  env
 ┃ ┃ ┗  warehouse_env.py          # Modélisation de l'entrepôt et des commandes
 ┃ ┣  metaheuristics
 ┃ ┃ ┣  aco_solver.py             # Algorithme des Colonies de Fourmis (ACO)
 ┃ ┃ ┣  genetic_solver.py         # Algorithme Génétique (AG)
 ┃ ┃ ┣  baseline.py               # Algorithme Glouton (Plus Proche Voisin)
 ┃ ┃ ┗  analyze_scaling.py        # Analyse des limites (Scalability)
 ┃ ┣  ml
 ┃ ┃ ┗  compare_algorithms.py     # Pipeline de clustering dynamique et évaluation
 ┃ ┣  main.py                     # Point d'entrée : Exécution du pipeline hybride
 ┃ ┗  compare_pipelines.py        # Point d'entrée : Comparaison globale (ACO vs AG)
 ┣  Template latex memoire M1     # Fichiers de rédaction du rapport final
 ┗  README.md                     # Documentation du projet
```

---

##  Prérequis et Installation

Assurez-vous d'avoir Python 3.8+ installé. Les bibliothèques requises pour le fonctionnement du Machine Learning et la génération des graphiques sont les suivantes :

```bash
# Mise à jour de pip
python -m pip install --upgrade pip

# Installation des dépendances
pip install numpy pandas matplotlib scikit-learn
```

---

##  Utilisation

### 1. Lancer le Pipeline Hybride (Recommandé)
Ce script exécute le scénario nominal : création de l'entrepôt, sélection du meilleur algorithme de clustering, découpage en lots, et calcul des tournées optimales avec les Colonies de Fourmis.

```bash
python src/main.py
```

### 2. Comparer les Algorithmes d'Optimisation
Ce script lance le grand affrontement entre les différentes approches (ACO Standard vs ACO Hybride vs Algorithme Génétique Hybride) pour démontrer l'apport du Machine Learning.

```bash
python src/compare_pipelines.py
```

### 3. Analyser la Scalabilité (Scaling Analysis)
Prouve l'effondrement des métaheuristiques seules sur de très grands volumes et justifie le besoin du clustering.

```bash
python src/metaheuristics/analyze_scaling.py
```

---

##  Fichiers de Sortie (Outputs)

Lors de l'exécution des scripts principaux, le module `ml` génère automatiquement des fichiers d'analyse :
* `fig1_comparaison_clusterings.png` : Vue spatiale des clusters selon les différents algorithmes.
* `fig2_metriques_comparaison.png` : Graphiques en barres des performances (Silhouette, Davies-Bouldin, Temps d'exécution).
* `fig3_best_clustering_detail.png` : Focus sur l'algorithme gagnant et la distribution des charges.
* `tableau_comparaison_ml.csv` : Données brutes des métriques d'évaluation.

---

##  Équipe du Projet

| Membre | Rôles & Contributions |
| :--- | :--- |
| **Lyna** | Modélisation de l'environnement (Warehouse Environment), Algorithme Glouton (Baseline), Algorithme Génétique & Intégration globale |
| **Kenza** | Comparaison de 4 algorithmes de clustering (K-Means, Agglomerative, GMM, DBSCAN), évaluation avec 3 métriques et génération des figures comparatives et export des résultats vers la métaheuristique|
| **Dyhia** | Implémentation des métaheuristiques (ACO et Algorithme Génétique) et comparaison des performances |

*Projet réalisé dans le cadre de la formation Master 1 MIAGE.*
