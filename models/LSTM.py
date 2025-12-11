# models/LSTM.py

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Último passo da sequência
        return out


class LSTMEngine:
    """
    Wrapper utilizado pelos agentes e pela LLM gerente.
    Permite treinar, prever e gerenciar o modelo LSTM.
    """

    def __init__(self, input_size, model_path="models/checkpoints/lstm_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)

        # Cria diretório se não existir
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = LSTMModel(input_size=input_size).to(self.device)

    def train(self, train_loader, epochs=20, lr=0.001):
        """
        Treina o modelo usando DataLoader com dados sequenciais.
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"[LSTM] Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

        self.save()

    def predict(self, sequence_tensor):
        """
        Recebe um tensor [1, seq_len, features] e retorna uma previsão.
        """
        self.model.eval()
        with torch.no_grad():
            sequence_tensor = sequence_tensor.to(self.device)
            prediction = self.model(sequence_tensor)
            return prediction.cpu().numpy().tolist()

    def save(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"[LSTM] Modelo salvo em: {self.model_path}")

    def load(self):
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"[LSTM] Modelo carregado de: {self.model_path}")
            return True
        else:
            print("[LSTM] Nenhum modelo salvo encontrado.")
            return False
