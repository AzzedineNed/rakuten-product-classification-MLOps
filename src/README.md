# Pipeline MLOps — Classification produits Rakuten

Modèle léger **TF-IDF + Régression Logistique** (27 catégories), servi via une API REST.

## Structure

| Fichier | Rôle |
|---|---|
| `config.py` | Chemins et hyperparamètres centralisés |
| `preprocessing.py` | Nettoyage du texte + fusion designation/description |
| `predict.py` | `TfidfPredictor` : charge le modèle et prédit (réutilisé par API + évaluation) |
| `train.py` | Entraînement + sauvegarde du modèle et des métriques |
| `evaluate.py` | Rapport de classification, F1, matrice de confusion |
| `api.py` | API FastAPI de prédiction |

Toutes les commandes se lancent **depuis la racine du projet**, environnement activé :

```bash
source venv_rakuten/bin/activate
```

## 1. Entraînement

```bash
python -m src.train                    # split 80/20 + métriques de validation
python -m src.train --full             # entraîne sur 100 % des données
python -m src.train --max-features 20000
```

Produit : `models/tfidf_vectorizer.pkl`, `models/logistic_regression.pkl`,
`output/metrics_logistic_regression.json`.

## 2. Évaluation

```bash
python -m src.evaluate                 # rapport + matrice de confusion
python -m src.evaluate --predict-test  # + prédictions sur X_test_update.csv
```

Produit : `figures/confusion_matrix_logreg.png` et le rapport détaillé en console.

Performances actuelles (validation) : **accuracy 0.78 · F1 weighted 0.78 · F1 macro 0.76**.

## 3. API

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Documentation interactive : http://localhost:8000/docs

```bash
# Prédiction unique
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"designation":"Console de jeux PS4","description":"manette incluse"}'
# → {"prdtypecode": 2462, "confidence": 0.37}

# Prédiction par lot
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"produits":[{"designation":"Livre roman policier"}]}'
```

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | GET | État de l'API et du modèle |
| `/predict` | POST | Prédiction pour un produit |
| `/predict/batch` | POST | Prédiction pour une liste de produits |
