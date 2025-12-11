# models/LSTM.py
# Modelo LSTM para previsão temporal de risco de quebra e estado futuro das máquinas.

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM para previsão temporal de:
        - risco de quebra (0 a 1)
        - estado futuro (regressão ou classificação)
    
    Entradas esperadas:
        x -> (batch, seq_len, input_size)

    Saídas:
        - estado: previsões do estado futuro (dimensão configurável)
        - risco_quebra: probabilidade 0..1
    """

    def __init__(self, input_size=4, hidden_size=64, num_layers=2, state_size=1):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Camada para prever estado futuro
        self.fc_state = nn.Linear(hidden_size, state_size)

        # Camada para probabilidade de quebra
        self.fc_risk = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x -> (batch, seq_len, input_size)
        """
        batch = x.size(0)

        h0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch, self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))

        # Pegando a última saída da sequência
        last_output = out[:, -1, :]

        estado = self.fc_state(last_output)
        risco = torch.sigmoid(self.fc_risk(last_output))

        return estado, risco


# ------------------------------------------------
# Função auxiliar para carregar modelo treinado
# ------------------------------------------------

def load_lstm_model(model_path: str, input_size=4, hidden_size=64, num_layers=2, state_size=1):
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        state_size=state_size
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model
