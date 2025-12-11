# models/randon_forest.py
# Wrapper para RandomForest de classificação/regressão de defeitos e gravidade.

import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np


class RandomForestEngine:
    """
    Motor de inferência Random Forest usado para:
      - Classificação de defeitos
      - Estimativa de gravidade
      - Previsão de probabilidade de falha

    `mode`:
        - "classification"
        - "regression"
    """

    def __init__(self, mode="classification", model_path=None):
        self.mode = mode
        self.model = None
        self.model_path = model_path or self.default_path()

    # ----------------------------------------------------
    # Path padrão para o modelo
    # ----------------------------------------------------
    def default_path(self):
        if self.mode == "classification":
            return "models/checkpoints/random_forest_class.pkl"
        else:
            return "models/checkpoints/random_forest_reg.pkl"

    # ----------------------------------------------------
    # Carregar modelo
    # ----------------------------------------------------
    def load(self):
        try:
            self.model = joblib.load(self.model_path)
            print(f"[RandomForest] Modelo carregado: {self.model_path}")
        except:
            print(f"[RandomForest] Nenhum modelo encontrado em {self.model_path}. Criando modelo novo.")

            # Modelo padrão para evitar crash
            if self.mode == "classification":
                self.model = RandomForestClassifier(n_estimators=50)
            else:
                self.model = RandomForestRegressor(n_estimators=50)

    # ----------------------------------------------------
    # Previsão
    # ----------------------------------------------------
    def predict(self, X):
        """
        Previsão simples (classe ou valor).
        X: lista de listas ou numpy array
        """
        X = np.array(X)
        return self.model.predict(X)

    # ----------------------------------------------------
    # Probabilidade da classificação
    # ----------------------------------------------------
    def predict_proba(self, X):
        """
        Apenas para classificação.
        Retorna vetor de probabilidades.
        """
        if self.mode != "classification":
            return None

        X = np.array(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    # ----------------------------------------------------
    # Salvar modelo
    # ----------------------------------------------------
    def save(self, path=None):
        path = path or self.model_path
        joblib.dump(self.model, path)
        print(f"[RandomForest] Modelo salvo em {path}")
