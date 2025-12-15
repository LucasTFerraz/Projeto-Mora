from LLM_G import LLMGerente
dados = [
        {"Maquina": "M1", "Estado": 0, "Defeito": -1},
        {"Maquina": "M2", "Estado": 1, "Defeito": 0},
        {"Maquina": "M3", "Estado": 2, "Defeito": 1}
      ]
llm_gerente = LLMGerente()
print("AA")
resposta = llm_gerente.handle(dados)
print(resposta)