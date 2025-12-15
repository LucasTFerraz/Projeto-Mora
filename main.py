# ---------------------------------------------------
# main.py
# API do Projeto MORA - Gerente Inteligente
# ---------------------------------------------------

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

#from models.randon_forest import RandomForestEngine
#from models.LSTM import LSTMEngine
from LLM_G import LLMGerente

# ---------------------------------------------------
# Inicialização da aplicação FastAPI
# ---------------------------------------------------

app = FastAPI(
    title="Projeto MORA - Gerente Inteligente",
    description="Sistema com LLM + Modelos de IA para suporte a decisões.",
    version="1.0.0",
)

# ---------------------------------------------------
# Inicialização dos motores de IA
# ---------------------------------------------------

'''# Random Forest
rf_engine = RandomForestEngine(mode="classification")
rf_engine.load()
'''
'''# LSTM temporal (4 features)
lstm_engine = LSTMEngine(input_size=4)
lstm_engine.load()'''

# Gerente LLM (ontologia + regras + modelos)
llm_gerente = LLMGerente()

# ---------------------------------------------------
# Modelos de requisição (Pydantic)
# ---------------------------------------------------

class RFRequest(BaseModel):
    data: List[List[float]]  # [[f1, f2, f3, f4], ...]

class LSTMRequest(BaseModel):
    sequence: List[List[float]]  # [[f1, f2, f3, f4], ...] ao longo do tempo

class MaquinaEstadoDefeito(BaseModel):
    Maquina: str
    Estado: int
    Defeito: int

class GerenteStructuredRequest(BaseModel):
    """
    Exemplo:
    {
      "maquinas": [
        {"Maquina": "M1", "Estado": 0, "Defeito": -1},
        {"Maquina": "M2", "Estado": 1, "Defeito": 0}
      ]
    }
    """
    maquinas: List[MaquinaEstadoDefeito]

class GerenteTextRequest(BaseModel):
    mensagem: str
    contexto_extra: Optional[Dict[str, Any]] = None

# ---------------------------------------------------
# Rotas básicas
# ---------------------------------------------------

@app.get("/")
def root():
    return {"msg": "API do Projeto MORA funcionando!"}

# ---------------------------------------------------
# LLM GERENTE - ENTRADA ESTRUTURADA
# ---------------------------------------------------

@app.post("/gerente")
def gerente_llm():
    """
    Entrada:
    {
      "maquinas": [
        {"Maquina": "M1", "Estado": 0, "Defeito": -1},
        {"Maquina": "M2", "Estado": 1, "Defeito": 0},
        {"Maquina": "M3", "Estado": 2, "Defeito": 1}
      ]
    }

    Saída (exemplo):
    {
      "resposta": "Maquina M1 Esta normal\nMaquina M2 ainda Esta longe De quebrar K e quebrar J, funcionário júnior C será enviado sozinho\nMaquina M3 Esta se aproximando de quebrar I, funcionário senior A vai ser enviado junto de funcionário treinando B"
    }
    """
    dados = [
        {"Maquina": "M1", "Estado": 0, "Defeito": -1},
        {"Maquina": "M2", "Estado": 1, "Defeito": 0},
        {"Maquina": "M3", "Estado": 2, "Defeito": 1}
      ]
    print("AA")
    resposta = llm_gerente.handle(dados)
    return {"resposta": resposta}

# ---------------------------------------------------
# LLM GERENTE - TEXTO LIVRE (opcional)
# ---------------------------------------------------

@app.post("/gerente_texto")
def gerente_llm_texto(req: GerenteTextRequest):
    """
    Entrada:
    {
      "mensagem": "M1 está fazendo barulho no eixo y",
      "contexto_extra": {...}  # opcional
    }

    Saída:
    {
      "resposta": "Máquina M1 está normal, funcionário C (júnior) será enviado sozinho."
    }
    """
    # usa o fluxo de texto do LLM_G.py
    resposta = llm_gerente.handle(req.mensagem)
    return {"resposta": resposta}

# ---------------------------------------------------
# Execução direta (uvicorn)
# ---------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
