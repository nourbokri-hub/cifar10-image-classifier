# CIFAR-10 Vision AI

Application de classification d’images CIFAR-10 réunissant un modèle PyTorch,
une API FastAPI et une interface Streamlit. Le projet couvre l’entraînement et
le suivi des expériences avec MLflow, ainsi que le lancement local ou avec
Docker Compose.

> **Checkpoint requis :** l’API et le conteneur API ont besoin du fichier
> `model.pt` à la racine du projet. Ce fichier est actuellement exclu de Git :
> il n’est donc pas présent dans un clone standard du dépôt. Placez le checkpoint
> fourni séparément à cet emplacement avant de lancer l’API ou Docker Compose.
> Le fichier n’est ni fourni ni recréé automatiquement par les commandes de
> démarrage.

## Résultats du modèle

Le checkpoint `model.pt` a été évalué sur les 10 000 images du jeu de test
CIFAR-10 :

| Mesure | Résultat |
|---|---:|
| Accuracy | **77,92 %** |
| Test loss | **0,6478** |
| Prédictions correctes | **7 792 / 10 000** |

Ces résultats correspondent au modèle existant. Les commandes de démarrage
ci-dessous ne réentraînent pas le modèle.

## Modèle et données

- **Jeu de données :** CIFAR-10, 60 000 images RGB de 32 × 32 pixels réparties
  en 10 classes (50 000 images d’entraînement et 10 000 de test).
- **Architecture :** ResNet18, adaptée à 10 classes et initialisée avec
  `weights=None`.
- **Entraînement :** réalisé à partir de zéro, sans poids pré-entraînés et sans
  Transfer Learning.
- **Prétraitement d’entraînement :** retournement horizontal et recadrage
  aléatoires, conversion en tenseur et normalisation RGB avec moyenne et
  écart-type de 0,5.
- **Prétraitement d’évaluation :** conversion en tenseur et même normalisation.
- **Optimiseur et perte :** Adam et `CrossEntropyLoss`.

Les transformations et le chargement de CIFAR-10 sont définis dans
[`src/dataset.py`](src/dataset.py); l’architecture est définie dans
[`src/model.py`](src/model.py).

## Technologies

- Python, PyTorch et Torchvision
- FastAPI et Uvicorn
- Streamlit
- MLflow
- Docker et Docker Compose
- Pytest

Les versions des paquets sont indiquées dans [`requirements.txt`](requirements.txt).

## Fonctionnalités

### API FastAPI

L’API charge `model.pt` au démarrage. Elle expose :

- `GET /health` : état de santé de l’API.
- `POST /predict` : reçoit une image PNG ou JPEG (maximum 10 Mio), la convertit
  en RGB, la redimensionne à 32 × 32 puis applique la normalisation utilisée
  pour l’évaluation.

La réponse de prédiction conserve les champs :

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

Les valeurs ci-dessus illustrent uniquement la forme de la réponse; elles ne
constituent pas une prédiction mesurée. La confiance est un score compris entre
0 et 1. L’interface Streamlit affiche les noms des classes en français.

### Interface Streamlit

L’interface permet de sélectionner une image JPG ou PNG, d’en afficher l’aperçu,
puis de demander une prédiction à l’API. Elle présente la classe prédite, son
score de confiance et un graphique horizontal des trois classes les plus
probables. L’adresse de l’API peut être configurée avec la variable
d’environnement `API_URL`.

### Suivi des expériences

Le script d’entraînement utilise MLflow pour enregistrer les paramètres et les
métriques par époque. L’interface MLflow locale peut être lancée avec
`mlflow ui` depuis la racine du dépôt.

## Installation et lancement local

Les commandes suivantes sont à exécuter depuis la racine du projet. Vérifiez
que `model.pt` s’y trouve avant de démarrer l’API.

### Environnement

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### API

```powershell
uvicorn api.main:app --reload
```

- API : <http://127.0.0.1:8000>
- Documentation interactive : <http://127.0.0.1:8000/docs>
- Contrôle de santé : <http://127.0.0.1:8000/health>

### Interface

Dans un second terminal, activez le même environnement puis lancez :

```powershell
streamlit run streamlit_app/app.py
```

L’interface est disponible par défaut sur <http://localhost:8501>. En exécution
locale, l’application utilise par défaut `http://127.0.0.1:8000/predict`.

### MLflow

```powershell
mlflow ui
```

L’interface est disponible par défaut sur <http://127.0.0.1:5000>.

### Tests

Depuis la racine du dépôt :

```powershell
pytest -v
```

Lors de la dernière exécution rapportée, **11 tests ont réussi**. Ce résultat
est un relevé d’exécution et ne garantit pas celui d’un environnement différent.

## Lancement avec Docker Compose

Docker Desktop (ou un moteur Docker compatible) doit être démarré. Le checkpoint
`model.pt` doit être présent à la racine avant la construction : le
[`Dockerfile de l’API`](docker/Dockerfile.api) le copie dans son image.

```powershell
docker compose up --build
```

Compose démarre l’API et attend que son contrôle de santé `/health` réussisse
avant de démarrer Streamlit. L’application utilise alors l’adresse interne
`http://api:8000/predict`.

- Interface Streamlit : <http://localhost:8501>
- API FastAPI : <http://localhost:8000>
- Documentation API : <http://localhost:8000/docs>

Pour arrêter les services, utilisez `Ctrl+C`, puis :

```powershell
docker compose down
```

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
├── README.md
└── requirements.txt
```

`model.pt`, le dossier `data/`, les fichiers MLflow (`mlruns/` et
`mlflow.db`) et l’environnement virtuel sont des éléments locaux ignorés par
Git; ils ne figurent pas dans cette arborescence des fichiers suivis.

## Limites connues

- Il n’existe pas de jeu de validation indépendant : le jeu de test CIFAR-10
  est utilisé pour calculer les métriques nommées `val_loss` et `val_acc` à
  chaque époque du script d’entraînement.
- Le dépôt ne fournit pas encore de matrice de confusion ni d’analyse
  détaillée des résultats par classe.
- `model.pt` est requis pour utiliser l’API et construire son image Docker, mais
  il est exclu de Git. Un clone du dépôt seul ne suffit donc pas pour lancer
  ces composants.
- Le script d’entraînement refuse de démarrer lorsqu’un `model.pt` existe déjà,
  afin d’éviter son écrasement accidentel. Aucune des commandes de démarrage
  documentées ci-dessus ne lance cet entraînement.

## Auteurs

- Nour Bokri
- Farah Jamai
- Cyrine Daghfous
## Télécharger le modèle entraîné

Le modèle ResNet18 entraîné est disponible dans les **GitHub Releases** du projet.

**[Télécharger le m odèle](https://github.com/nourbokri-hub/cifar10-image-classifier/releases/download/v1.0.0/model.pt)[`model.pt`](https://github.com/nourbokri-hub/cifar10-image-classifier/releases/download/v1.0.0/model.pt)**

### Installation du modèle

1. Téléchargez le fichier `model.pt` depuis le lien ci-dessus.
2. Placez-le à la racine du projet, au même niveau que `requirements.txt` et `docker-compose.yml`.
3. Lancez l'application en suivant les instructions d'installation.

Le modèle a obtenu une précision de **77,92 %** sur les 10 000 images du jeu de test CIFAR-10.

Le fichier `model.pt` n'est pas suivi directement par Git afin d'éviter d'alourdir le dépôt.
