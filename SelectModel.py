import traceback
from models.CNN import CNNModel
from models.LSTM import LSTMModel
from models.random_forest import RandomForestModel


class ModelSelector:
    """
    Controla qual modelo será usado para prever:
    - tempo até falha
    - chance de defeito
    - gravidade
    """

    def __init__(self):
        # Lista de modelos na ordem de prioridade
        self.models = [
            ("LSTM", LSTMModel),
            ("CNN", CNNModel),
            ("RandomForest", RandomForestModel)
        ]

        self.loaded_models = {}
        self.load_all()

    # ------------------------------------------------------------------
    # CARREGAR TODOS OS MODELOS DISPONÍVEIS
    # ------------------------------------------------------------------
    def load_all(self):
        for name, ModelClass in self.models:
            try:
                model_instance = ModelClass()
                model_instance.load()

                self.loaded_models[name] = model_instance
                print(f"[ModelSelector] Modelo carregado: {name}")

            except Exception:
                print(f"[ModelSelector] Erro ao carregar {name}:")
                traceback.print_exc()

        if not self.loaded_models:
            raise RuntimeError("Nenhum modelo de IA pôde ser carregado.")

    # ------------------------------------------------------------------
    # ESCOLHER O MELHOR MODELO AUTOMATICAMENTE
    # ------------------------------------------------------------------
    def select_best(self):
        """
        Seleciona o melhor modelo disponível seguindo a ordem de prioridade:
        1. LSTM (melhor para séries temporais)
        2. CNN
        3. RandomForest (fallback)
        """
        for name, _ in self.models:
            if name in self.loaded_models:
                return self.loaded_models[name]

        # nunca deveria chegar aqui
        raise RuntimeError("Erro inesperado: nenhum modelo carregado.")

    # ------------------------------------------------------------------
    # PREDIÇÃO UNIFICADA
    # ------------------------------------------------------------------
    def predict(self, data):
        """
        Aplica o melhor modelo.
        'data' deve ser um vetor ou dict pré-processado.
        O retorno deve ser padronizado como:

        {
            "time_to_fail": 12.4,          # horas
            "severity": 0.8,               # 0 a 1
            "failure_type": "bearing_wear"
        }
        """

        model = self.select_best()

        try:
            result = model.predict(data)
            return result
        except Exception:
            # fallback automático
            print(f"[ModelSelector] Erro com {model}. Tentando fallback...")
            traceback.print_exc()

            return self.predict_with_fallback(data, ignore=model)

    # ------------------------------------------------------------------
    # FALLBACK: SE UM MODELO FALHAR, OUTRO ENTRA
    # ------------------------------------------------------------------
    def predict_with_fallback(self, data, ignore=None):
        for name, ModelClass in self.models:
            if name in self.loaded_models and self.loaded_models[name] is not ignore:
                try:
                    print(f"[ModelSelector] Fallback → {name}")
                    return self.loaded_models[name].predict(data)
                except Exception:
                    continue
        raise RuntimeError("Todos os modelos falharam.")

