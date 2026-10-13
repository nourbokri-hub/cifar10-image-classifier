# CIFAR-10 Vision AI

Application de classification d'images développée avec **PyTorch, ResNet18, FastAPI et Streamlit**.

Ce projet a pour objectif de concevoir, entraîner, évaluer et exploiter un modèle de Deep Learning capable de reconnaître automatiquement 10 catégories d'images du jeu de données CIFAR-10.

Il couvre les principales étapes d'un pipeline de Machine Learning : préparation des données, entraînement, évaluation, suivi des expériences avec MLflow et utilisation du modèle à travers une API et une interface web.

Le projet peut être exécuté localement ou à l'aide de Docker Compose.

## Résultats du modèle

Le modèle ResNet18 enregistré dans `model.pt` a été évalué sur les 10 000 images du jeu de test CIFAR-10.

| Métrique | Résultat |
|---|---|
| Architecture | ResNet18 |
| Précision (Accuracy) | **77,92 %** |
| Perte sur le jeu de test | **0,6478** |
| Images correctement classées | **7 792 / 10 000** |
| Nombre de classes | 10 |

Ces résultats ont été vérifiés en chargeant le modèle entraîné et en l'évaluant sur le jeu de test CIFAR-10, sans effectuer de nouvel entraînement.

## Modèle et données

### Jeu de données CIFAR-10

Le projet utilise le dataset CIFAR-10, composé de 60 000 images couleur RGB de dimensions 32 × 32 pixels :

- 50 000 images d'entraînement
- 10 000 images de test
- 10 catégories : avion, automobile, oiseau, chat, cerf, chien, grenouille, cheval, bateau et camion

### Architecture ResNet18

Le modèle repose sur ResNet18, un réseau de neurones convolutif utilisant des connexions résiduelles.

La dernière couche entièrement connectée a été adaptée pour prédire les 10 classes de CIFAR-10.

**Configuration :**
- Framework : PyTorch
- Architecture : ResNet18
- Initialisation : `weights=None`
- Optimiseur : Adam
- Fonction de perte : `CrossEntropyLoss`
- Taux d'apprentissage par défaut : 0,001
- Taille des lots par défaut : 64

Le modèle est entraîné à partir de zéro, sans poids pré-entraînés et sans Transfer Learning.

### Prétraitement des images

**Pendant l'entraînement :**
- Retournement horizontal aléatoire
- Recadrage aléatoire avec padding
- Conversion des images en tenseurs
- Normalisation RGB avec une moyenne et un écart-type de 0,5

**Pendant l'évaluation :**
- Conversion en tenseurs
- Application de la même normalisation RGB

Le chargement des données et les transformations sont définis dans [src/dataset.py](src/dataset.py).

L'architecture du modèle est définie dans [src/model.py](src/model.py).

## Technologies utilisées

| Domaine | Technologies |
|---|---|
| Deep Learning | Python, PyTorch, Torchvision |
| Computer Vision | ResNet18, CIFAR-10 |
| API REST | FastAPI, Uvicorn |
| Interface utilisateur | Streamlit |
| Suivi des expériences | MLflow |
| Conteneurisation | Docker, Docker Compose |
| Tests automatisés | Pytest |

Les dépendances Python sont indiquées dans [requirements.txt](requirements.txt).

## Fonctionnalités

### API FastAPI

L'API charge le modèle entraîné depuis `model.pt` au démarrage.

Elle propose deux endpoints :

- `GET /health` : vérification de l'état de santé de l'API.
- `POST /predict` : classification d'une image envoyée par l'utilisateur.

L'API accepte les images PNG et JPEG, avec une taille maximale de 10 Mio.

Les images sont converties en RGB, redimensionnées à 32 × 32 pixels, puis normalisées avant leur passage dans ResNet18.

**Exemple de structure d'une réponse JSON :**

```json
{
  "prediction": "cat",
  "confidence": 0.72,
  "top3": [
    {"class": "cat", "confidence": 0.72},
    {"class": "dog", "confidence": 0.18},
    {"class": "frog", "confidence": 0.07}
  ]
}
```

Les valeurs de cet exemple sont fictives et servent uniquement à illustrer la structure de la réponse.

Les scores de confiance sont compris entre 0 et 1.

### Interface Streamlit

Le projet comprend une interface web développée avec Streamlit, intitulée **CIFAR-10 Vision AI**.

L'interface est disponible en français et utilise un thème clair violet/lilas.

Elle permet de :

- Importer une image JPG ou PNG.
- Visualiser l'image sélectionnée.
- Lancer une prédiction à partir de l'API FastAPI.
- Afficher la catégorie prédite.
- Consulter le score de confiance.
- Visualiser les trois catégories les plus probables dans un graphique horizontal.

L'interface conserve le résultat de la prédiction tant que l'image importée ne change pas.

L'adresse de l'API peut être configurée à l'aide de la variable d'environnement `API_URL`.

### Suivi des expériences avec MLflow

Le script d'entraînement intègre MLflow pour enregistrer les paramètres et les performances du modèle.

**Paramètres suivis :**
- Nombre d'époques
- Taille des lots
- Taux d'apprentissage
- Dispositif de calcul (CPU ou CUDA)

**Métriques suivies :**
- Perte d'entraînement
- Précision d'entraînement
- Perte d'évaluation
- Précision d'évaluation
- Précision finale

Les poids entraînés sont sauvegardés dans `model.pt` et enregistrés comme artefact MLflow.

## Télécharger le modèle entraîné

Le fichier `model.pt` contient les poids du modèle ResNet18 entraîné.

Il est disponible dans la **GitHub Release v1.0.0** du projet.

**[📥 Télécharger le modèle entraîné (model.pt)](https://github.com/nourbokri-hub/cifar10-image-classifier/releases/download/v1.0.0/model.pt)**

### Installation du modèle

1. Téléchargez `model.pt` depuis le lien ci-dessus.
2. Placez le fichier à la racine du projet, au même niveau que `requirements.txt` et `docker-compose.yml`.
3. Lancez l'application en suivant les instructions ci-dessous.

**Important :** `model.pt` est exclu du suivi Git afin de ne pas alourdir le dépôt. Son téléchargement est donc nécessaire après un nouveau clonage du projet.

Le fichier n'est pas téléchargé automatiquement lors du démarrage de l'application.

## Installation et lancement local

Les commandes suivantes doivent être exécutées depuis la racine du projet.

### 1. Cloner le dépôt

```powershell
git clone https://github.com/nourbokri-hub/cifar10-image-classifier.git
cd cifar10-image-classifier
```

### 2. Créer l'environnement virtuel

```powershell
python -m venv .venv
```

Activer l'environnement sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```powershell
python -m pip install -r requirements.txt
```

### 4. Ajouter le modèle

Téléchargez `model.pt` depuis la GitHub Release et placez-le à la racine du projet.

Le fichier est indispensable au démarrage de l'API.

### 5. Lancer FastAPI

```powershell
uvicorn api.main:app --reload
```

Accès :

- API : http://127.0.0.1:8000
- Documentation Swagger : http://127.0.0.1:8000/docs
- Contrôle de santé : http://127.0.0.1:8000/health

### 6. Lancer Streamlit

Dans un deuxième terminal, activez le même environnement virtuel, puis exécutez :

```powershell
streamlit run streamlit_app/app.py
```

L'interface est disponible à l'adresse :

http://localhost:8501

En exécution locale, Streamlit utilise par défaut l'endpoint `http://127.0.0.1:8000/predict`.

### 7. Consulter MLflow

```powershell
mlflow ui
```

Interface MLflow :

http://127.0.0.1:5000

## Lancement avec Docker Compose

Docker permet de lancer les composants applicatifs dans des conteneurs.

Le fichier `docker-compose.yml` définit deux services :

- **api** : backend FastAPI
- **streamlit** : interface utilisateur

### Démarrage

Assurez-vous que Docker Desktop fonctionne et que `model.pt` se trouve à la racine du projet.

Exécutez :

```powershell
docker compose up --build -d
```

Docker construit les images et démarre les deux services.

Le service Streamlit utilise l'adresse interne `http://api:8000/predict` pour communiquer avec FastAPI.

Un contrôle de santé vérifie la disponibilité de l'API avant le démarrage du service Streamlit.

### Accès à l'application

- Streamlit : http://localhost:8501
- FastAPI : http://localhost:8000
- Documentation API : http://localhost:8000/docs

### Vérification des conteneurs

```powershell
docker compose ps
```

### Arrêt des services

```powershell
docker compose down
```

## Tests automatisés

Le projet utilise Pytest pour vérifier plusieurs composants.

Les tests portent notamment sur :

- La création du modèle ResNet18.
- La forme de ses sorties.
- La disponibilité des routes FastAPI.
- Les prédictions sur des images PNG et JPEG.
- Le rejet des images invalides.
- Le rejet des formats non pris en charge.
- Le rejet des images dépassant la taille autorisée.
- La protection du checkpoint existant contre l'écrasement.

### Exécution des tests

```powershell
python -m pytest -v
```

**Dernier résultat vérifié : 11 tests réussis sur 11.**

Ce résultat correspond à l'environnement de développement utilisé et ne garantit pas automatiquement le même résultat sur toutes les configurations.

## Structure du dépôt

```text
cifar10-image-classifier/
├── api/
│   ├── __init__.py
│   └── main.py
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.streamlit
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── model.py
│   └── training.py
├── streamlit_app/
│   └── app.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_model.py
│   └── test_predict.py
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Les éléments suivants sont utilisés localement et exclus du suivi Git :

- `model.pt` : poids du modèle entraîné
- `data/` : jeu de données CIFAR-10
- `mlruns/` et `mlflow.db` : données de suivi MLflow
- `.venv/` : environnement virtuel Python

## Limites du projet

Le projet présente certaines limites :

- Aucun jeu de validation indépendant n'est utilisé pendant l'entraînement actuel.
- Les métriques `val_loss` et `val_acc` sont calculées sur le jeu de test CIFAR-10 après chaque époque.
- Aucune matrice de confusion n'est encore fournie.
- Les performances détaillées par classe ne sont pas disponibles.
- Le fichier `model.pt` doit être téléchargé séparément pour exécuter les prédictions.

Le script d'entraînement comporte également une protection empêchant l'écrasement accidentel d'un checkpoint existant.

## Auteurs

- Nour Bokri
- Farah Jamai
- Cyrine Daghfous

## Conclusion

Ce projet présente une application de classification d'images basée sur un modèle **ResNet18 entraîné avec PyTorch**, associée à une API FastAPI et à une interface Streamlit.

Le modèle a obtenu une précision de **77,92 % sur le jeu de test CIFAR-10**, composé de 10 000 images.

L'intégration de MLflow, Docker et Pytest permet également d'aborder différentes pratiques liées au développement, au suivi des expériences, aux tests et à l'exécution d'applications de Machine Learning.