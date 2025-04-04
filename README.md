# TrainSense

TrainSense est un package Python ultra puissant et robuste conçu pour analyser en profondeur l'architecture de modèles, optimiser les hyperparamètres en fonction de la configuration système et effectuer des diagnostics avancés. Il offre une analyse complète, allant du profilage du modèle aux diagnostics système, en passant par l'ajustement automatique des hyperparamètres.

---

## Table des Matières

- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Configuration du Projet](#configuration-du-projet)
- [Utilisation](#utilisation)
  - [Exemple Basique](#exemple-basique)
  - [Rapport d'Analyse Profonde](#rapport-danalyse-profonde)
- [Dépendances](#dépendances)
- [Licence](#licence)

---

## Fonctionnalités

- **Analyse des Hyperparamètres**  
  Vérifie et ajuste les hyperparamètres (batch_size, learning_rate, epochs) en fonction de la configuration système et des caractéristiques du modèle.
- **Analyse de l'Architecture**  
  Calcule le nombre total de paramètres et de couches, et fournit des recommandations sur l'architecture.
- **Analyse Profonde**  
  Agrège les résultats d'analyse hyperparamétrique, de profilage du modèle et de diagnostics système pour fournir un rapport complet.
- **Profilage du Modèle**  
  Mesure la performance en termes de temps d'inférence, débit et utilisation de la mémoire.
- **Diagnostics Système**  
  Recueille des informations sur l'utilisation CPU, mémoire, disque et autres caractéristiques système.
- **Optimisation Ultra**  
  Propose des hyperparamètres optimaux basés sur les statistiques des données, l'architecture du modèle et les ressources système.

---

## Installation

### Prérequis

- Python 3.6+
- [pip](https://pip.pypa.io/)

### Étapes d'Installation

1. **Cloner le dépôt GitHub :**

   ```bash
   git clone https://github.com/toncompte/TrainSense.git
   cd TrainSense
   ```

2. **Installer en mode développement :**

   ```bash
   pip install -e .
   ```

3. **Installer les dépendances (si elles ne sont pas déjà installées) :**

   Le package nécessite les modules suivants :
   - GPUtil
   - psutil
   - torch

   Si nécessaire, installez-les via pip :

   ```bash
   pip install GPUtil psutil torch
   ```

---

## Configuration du Projet

Le projet est organisé comme suit :

```
TrainSense/
├── TrainSense/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── arch_analyzer.py
│   ├── deep_analyzer.py
│   ├── gpu_monitor.py
│   ├── logger.py
│   ├── model_profiler.py
│   ├── optimizer.py
│   ├── ultra_optimizer.py
│   ├── system_config.py
│   ├── system_diagnostics.py
│   └── utils.py
├── examples/
│   └── example_usage.py
└── setup.py
```

Chaque module a une responsabilité spécifique pour fournir une analyse complète et des recommandations d'optimisation.

---

## Utilisation

### Exemple Basique

Vous pouvez utiliser TrainSense pour obtenir un résumé de la configuration, vérifier les hyperparamètres et obtenir des recommandations d'optimisation.

```python
from TrainSense.system_config import SystemConfig
from TrainSense.analyzer import TrainingAnalyzer
from TrainSense.arch_analyzer import ArchitectureAnalyzer
from TrainSense.logger import TrainLogger
import torch
import torch.nn as nn
from TrainSense.utils import print_section

# Initialisation de la configuration système
config = SystemConfig()

# Définition d'un modèle de test avec PyTorch
model = nn.Sequential(nn.Linear(10, 256), nn.ReLU(), nn.Linear(256, 10))
arch_analyzer = ArchitectureAnalyzer(model)
arch_info = arch_analyzer.analyze()

# Initialisation des hyperparamètres
batch_size = 64
learning_rate = 0.05
epochs = 30

# Création de l'analyseur de formation
analyzer = TrainingAnalyzer(batch_size, learning_rate, epochs, system_config=config, arch_info=arch_info)

# Affichage du résumé de la configuration
print_section("Résumé de la configuration")
summary = analyzer.summary()
for key, value in summary.items():
    print(f"{key}: {value}")

# Vérification des hyperparamètres et ajustements proposés
print_section("Recommandations d'hyperparamètres")
for rec in analyzer.check_hyperparams():
    print(rec)
adjustments = analyzer.auto_adjust()
print_section("Ajustements automatiques proposés")
for key, value in adjustments.items():
    print(f"{key}: {value}")

# Journalisation (logs)
logger = TrainLogger(log_file="logs/trainsense.log")
logger.log_info("Analyse complète démarrée.")
```

### Rapport d'Analyse Profonde

Utilisez le module DeepAnalyzer pour générer un rapport complet regroupant l'analyse hyperparamétrique, le profilage du modèle et les diagnostics système.

```python
from TrainSense.system_config import SystemConfig
from TrainSense.system_diagnostics import SystemDiagnostics
from TrainSense.analyzer import TrainingAnalyzer
from TrainSense.arch_analyzer import ArchitectureAnalyzer
from TrainSense.deep_analyzer import DeepAnalyzer
from TrainSense.model_profiler import ModelProfiler
from TrainSense.optimizer import OptimizerHelper
from TrainSense.ultra_optimizer import UltraOptimizer
from TrainSense.utils import print_section
import torch
import torch.nn as nn

# Configuration système et diagnostics
sys_config = SystemConfig()
sys_diag = SystemDiagnostics()

# Création d'un modèle de test
model = nn.Sequential(nn.Linear(10, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
arch_analyzer = ArchitectureAnalyzer(model)
arch_info = arch_analyzer.analyze()

# Analyse des hyperparamètres
batch_size = 64
learning_rate = 0.05
epochs = 30
analyzer = TrainingAnalyzer(batch_size, learning_rate, epochs, system_config=sys_config, arch_info=arch_info)

# Profilage du modèle
profiler = ModelProfiler(model, device="cpu")

# Ultra optimisation basée sur la statistique des données (exemple fictif)
ultra_opt = UltraOptimizer({"data_size": 500000}, arch_info, {"total_memory_gb": sys_config.total_memory})

# Création d'un analyseur profond
deep_analyzer = DeepAnalyzer(analyzer, arch_analyzer, profiler, sys_diag)

# Génération du rapport complet
print_section("Rapport Complet d'Analyse Profonde")
report = deep_analyzer.comprehensive_report()
for key, value in report.items():
    print(f"{key}: {value}")

# Ajustement du learning rate basé sur la performance mesurée
new_lr, tune_msg = OptimizerHelper.adjust_learning_rate(learning_rate, report["profiling"]["throughput"])
print_section("Ajustement du Learning Rate Basé sur Performance")
print("Nouveau learning rate:", new_lr, "-", tune_msg)
```

---

## Dépendances

- [GPUtil](https://pypi.org/project/GPUtil/)
- [psutil](https://pypi.org/project/psutil/)
- [PyTorch](https://pytorch.org/)

Assurez-vous d'installer toutes les dépendances nécessaires pour garantir le bon fonctionnement du package.

---

## Licence

Ce projet est sous licence [MIT](LICENSE).

Profitez de TrainSense pour optimiser vos entraînements et obtenir des analyses ultra poussées de vos modèles et de votre configuration système !
