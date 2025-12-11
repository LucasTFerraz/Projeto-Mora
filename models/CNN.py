# models/CNN.py
# Modelo CNN 1D para previsão de defeitos e probabilidade de quebra em máquinas

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
    Modelo de CNN (1D) simples para classificação de defeitos
    e regressão da probabilidade de quebra.

    Entradas esperadas:
        - Sequência temporal de sensores: (batch, channels=1, seq_len)

    Saídas:
        - classe_defeito: logits para classificação de defeitos
        - prob_quebra: valor entre 0-1
    """

    def __init__(self, num_defeitos: int, seq_len: int = 100):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(2)

        # Cálculo do tamanho após pooling
        reduced_len = seq_len // 2 // 2  # após 2 poolings

        self.fc_defeito = nn.Linear(64 * reduced_len, num_defeitos)
        self.fc_quebra = nn.Linear(64 * reduced_len, 1)

    def forward(self, x):
        """
        x -> (batch, 1, seq_len)
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Saída da classificação (defeito)
        classe_defeito = self.fc_defeito(x)

        # Saída da probabilidade de quebra
        prob_quebra = torch.sigmoid(self.fc_quebra(x))

        return classe_defeito, prob_quebra


# -------------------------------
# Função auxiliar para carregar modelo
# -------------------------------

def load_cnn_model(model_path: str, num_defeitos: int):
    """
    Carrega o modelo em modo de inferência.
    """
    model = CNNModel(num_defeitos=num_defeitos)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model
