# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# Importando motores dos modelos
'''from models.randon_forest import RandomForestEngine
from models.LSTM import LSTMEngine'''
from LLM_G import LLMGerente


# Inicializa FastAPI
app = FastAPI(
    title="Projeto MORA - Gerente Inteligente",
    description="Sistema com LLM + Modelos de IA para suporte a decisões.",
    version="1.0.0"
)


# ---------------------------------------------------
# Carrega os modelos quando o servidor inicia
# ---------------------------------------------------

#rf_engine = RandomForestEngine(mode="classification")
#rf_engine.load()

#lstm_engine = LSTMEngine(input_size=4)  # Exemplo: 4 features
#lstm_engine.load()

llm_gerente = LLMGerente(
    rf_engine=rf_engine,
    lstm_engine=lstm_engine
)


# ---------------------------------------------------
# Schemas para entrada da API
# ---------------------------------------------------

class RFRequest(BaseModel):
    data: List[List[float]]


class LSTMRequest(BaseModel):
    sequence: List[List[float]]  # sequência temporal


class GerenteRequest(BaseModel):
    mensagem: str
    contexto_extra: Optional[dict] = None


# ---------------------------------------------------
# ROTAS DO SISTEMA
# ---------------------------------------------------


@app.get("/")
def root():
    return {"msg": "API do Projeto MORA funcionando!"}


'''# ---------- RANDOM FOREST ----------
@app.post("/predict/random_forest")
def rf_predict(req: RFRequest):
    pred = rf_engine.predict(req.data)
    return {"predictions": pred}


@app.post("/predict/random_forest_proba")
def rf_predict_proba(req: RFRequest):
    pred = rf_engine.predict_proba(req.data)
    return {"probabilities": pred}


# ---------- LSTM ----------
@app.post("/predict/lstm")
def lstm_predict(req: LSTMRequest):
    import torch
    tensor = torch.tensor([req.sequence], dtype=torch.float32)  # [1, seq_len, features]
    pred = lstm_engine.predict(tensor)
    return {"prediction": pred}
'''

# ---------- LLM GERENTE ----------
@app.post("/gerente")
def gerente_llm(req: GerenteRequest):
    resposta = llm_gerente.processar(req.mensagem, req.contexto_extra)
    return {"resposta": resposta}


# ---------------------------------------------------
# Inicialização
# ---------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
