# ---------------------------------------------------------------
# FORM_TEXT_INPUT.py
# Projeto-Mora — Processamento inicial de texto do operador
# ---------------------------------------------------------------

import re

class FormTextInput:

    def __init__(self):
        # palavras que podem indicar características
        self.map_caracteristicas = {
            "barulho": "Barulho",
            "ruido": "Barulho",
            "barulhento": "Barulho",

            "vibrando": "Danos",
            "vibração": "Danos",
            "tremendo": "Danos",

            "parado": "Funcional",
            "sem funcionar": "Funcional",
            "travado": "Funcional",

            "etc": "ETC"
        }

        # possíveis causas
        self.map_causas = {
            "eixo x": "Eixo_X",
            "eixo y": "Eixo_Y",
            "eixo z": "Eixo_Z",
            "s_i": "S_i",
            "s_j": "S_j",
            "s_k": "S_k"
        }

    # -----------------------------------------------------------
    # Função principal
    # -----------------------------------------------------------
    def parse(self, text_input: str) -> dict:
        """
        Recebe texto livre do operador e retorna um dicionário estruturado:
        {
           "maquina": "M1",
           "caracteristica": "Barulho",
           "causa": "Eixo_Y"
        }
        """

        text = text_input.lower().strip()

        # ----------------------------
        # 1) Identificação da máquina
        # ----------------------------
        maquina = None
        maquina_match = re.search(r"\b(m\d+|n\d+)\b", text)
        if maquina_match:
            maquina = maquina_match.group(1).upper()

        # ----------------------------
        # 2) Característica do defeito
        # ----------------------------
        caracteristica = None
        for palavra, classe in self.map_caracteristicas.items():
            if palavra in text:
                caracteristica = classe
                break

        # ----------------------------
        # 3) Possível causa
        # ----------------------------
        causa = None
        for palavra, classe in self.map_causas.items():
            if palavra in text:
                causa = classe
                break

        # Caso nenhuma causa seja encontrada, marcar como indefinida
        if causa is None:
            causa = "Desconhecida"

        # ----------------------------
        # 4) Construção da resposta
        # ----------------------------
        resultado = {
            "maquina": maquina,
            "caracteristica": caracteristica,
            "causa": causa,
            "texto_original": text_input
        }

        return resultado
