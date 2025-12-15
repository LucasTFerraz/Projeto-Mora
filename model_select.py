# model_select.py
"""
Gerenciador central de modelos do Projeto MORA.
Permite carregar, armazenar, consultar e executar modelos IA
(Randon Forest, LSTM etc.).
"""

from models.randon_forest import RandomForestEngine
#from models.LSTM import LSTMEngine


class ModelSelector:
    def __init__(self):
        self.models = {}
        self._load_all()

    # ----------------------------------------------------------
    # Carregar modelos disponíveis
    # ----------------------------------------------------------
    def _load_all(self):
        # Random Forest
        rf = RandomForestEngine(mode="classification")
        rf.load()
        self.models["rf"] = rf

        # LSTM
        '''lstm = LSTMEngine(input_size=4)
        lstm.load()
        self.models["lstm"] = lstm'''

        # Novos modelos poderão ser adicionados aqui
        # ex: self.models["cnn"] = CNNEngine(...)

    # ----------------------------------------------------------
    # Obter modelo pelo nome
    # ----------------------------------------------------------
    def get(self, model_name: str):
        """
        Retorna o modelo carregado.
        """
        return self.models.get(model_name)

    # ----------------------------------------------------------
    # Execução padronizada de previsão
    # ----------------------------------------------------------
    def predict(self, model_name: str, data):
        """
        Chama a previsão de um modelo específico.

        model_name:
            "rf"   -> RandomForest
            "lstm" -> LSTM temporal

        data:
            RF: lista de listas
            LSTM: tensor ou lista sequencial
        """
        if model_name not in self.models:
            return None

        model = self.models[model_name]

        # Random Forest
        if model_name == "rf":
            return model.predict(data)

        # LSTM
        if model_name == "lstm":
            import torch
            tensor = torch.tensor([data], dtype=torch.float32)
            return model.predict(tensor)

        return None

    # ----------------------------------------------------------
    # Execução de probabilidade (apenas classificação)
    # ----------------------------------------------------------
    def predict_proba(self, model_name: str, data):
        if model_name != "rf":
            return None
        return self.models["rf"].predict_proba(data)


# Instância global que pode ser importada onde quiser
model_select = ModelSelector()
