# models/randon_forest.py

import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RandomForestEngine:
    """
    Engine genérica para classificação ou regressão usando RandomForest.
    Usada pela LLM gerente e pelos agentes.
    """

    def __init__(self, mode="classification", model_path="models/checkpoints/random_forest.pkl", **kwargs):
        """
        mode: "classification" ou "regression"
        kwargs: hiperparâmetros do RandomForest
        """

        self.mode = mode
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "classification":
            self.model = RandomForestClassifier(**kwargs)
        else:
            self.model = RandomForestRegressor(**kwargs)

    def train(self, X_train, y_train):
        """
        Treina o modelo com arrays numpy ou pandas.
        """
        print("[RF] Treinando modelo RandomForest...")
        self.model.fit(X_train, y_train)
        self.save()
        print("[RF] Treinamento concluído!")

    def predict(self, X):
        """
        Realiza previsões.
        """
        return self.model.predict(X).tolist()

    def predict_proba(self, X):
        """
        Retorna probabilidades (apenas para classificação).
        """
        if self.mode == "classification" and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X).tolist()
        else:
            print("[RF] predict_proba não disponível para este modelo.")
            return None

    def save(self):
        """
        Salva o modelo em disco.
        """
        joblib.dump(self.model, self.model_path)
        print(f"[RF] Modelo salvo em: {self.model_path}")

    def load(self):
        """
        Carrega o modelo salvo.
        """
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            print(f"[RF] Modelo carregado de: {self.model_path}")
            return True
        else:
            print("[RF] Nenhum modelo salvo encontrado.")
            return False
