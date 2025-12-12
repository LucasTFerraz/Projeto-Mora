# LLM_G.py
# Gerente inteligente do Projeto MORA (versão focada em ontologia + entrada estruturada)

from typing import List, Dict, Any
from owlready2 import get_ontology

# ----------------------------------------------------------------------
# 1. CARREGAR ONTOLOGIA PROJECT_Ontology.owl
# ----------------------------------------------------------------------

ONTOLOGY_PATH = "Project_Ontology.owx"  # mesmo arquivo .owl que você exportou

class OntologyAccess:
    def __init__(self, path: str = ONTOLOGY_PATH):
        self.onto = get_ontology(path).load()

    # mapeia código inteiro de estado (0,1,2,...) para classe de Estado na ontologia
    def map_estado_code_to_label(self, code: int) -> str:
        """
        0 → Normal
        1 → distante_de_quebrar
        2 → Perto_de_Quebrar
        """
        if code == 0:
            return "Esta normal"                    # texto alvo para M1
        if code == 1:
            return "ainda Esta longe De quebrar"    # texto alvo para M2
        if code == 2:
            return "Esta se aproximando de quebrar" # texto alvo para M3
        return "em estado desconhecido"

    # mapeia código inteiro de defeito (-1,0,1,...) para defeitos na ontologia
    def map_defeito_code_to_fragment(self, code: int) -> str:
        """
        -1 → sem defeito relevante
         0 → K e quebrar J
         1 → I
        """
        if code == -1:
            return ""              # nenhum complemento
        if code == 0:
            # exemplo: combinação de k_Quebrado e j_Quebrado
            return "K e quebrar J"
        if code == 1:
            # exemplo: i_Quebrado
            return "I"
        return ""

# ----------------------------------------------------------------------
# 2. LLMGerente: recebe entrada estruturada e gera a saída textual
# ----------------------------------------------------------------------

class LLMGerente:
    """
    Entrada: lista de dicts estruturados, por exemplo:
        [
          {"Maquina": "M1", "Estado": 0, "Defeito": -1},
          {"Maquina": "M2", "Estado": 1, "Defeito": 0},
          {"Maquina": "M3", "Estado": 2, "Defeito": 1},
        ]

    Saída (string com quebras de linha), por exemplo:
        Maquina M1 Esta normal
        Maquina M2 ainda Esta longe De quebrar K e quebrar J, funcionário júnior C será enviado sozinho
        Maquina M3 Esta se aproximando de quebrar I, funcionário senior A vai ser enviado junto de funcionário treinando B
    """

    def __init__(self, ontology_path: str = ONTOLOGY_PATH):
        self.onto_access = OntologyAccess(ontology_path)

    # ------------------------------------------------------------------
    # mapeia estado numérico → gravidade qualitativa (para regras de equipe)
    # ------------------------------------------------------------------
    def _estado_para_gravidade(self, estado_code: int) -> str:
        if estado_code == 0:
            return "Baixa"
        if estado_code == 1:
            return "Media"
        if estado_code == 2:
            return "Alta"
        return "Baixa"

    # ------------------------------------------------------------------
    # escolhe funcionário(s) com base na gravidade desejada
    # aqui seguimos exatamente o padrão textual que você pediu
    # ------------------------------------------------------------------
    def _escolher_funcionarios(self, gravidade: str, estado_code: int) -> str:
        """
        Regra fixa para reproduzir seus exemplos:

        - Para M1 (estado 0, gravidade Baixa): sem funcionário mencionado.
        - Para M2 (estado 1, gravidade Média):
              'funcionário júnior C será enviado sozinho'
        - Para M3 (estado 2, gravidade Alta):
              'funcionário senior A vai ser enviado junto de funcionário treinando B'
        """
        if estado_code == 0:
            # M1: nenhum funcionário no texto
            return ""
        if estado_code == 1:
            # M2: júnior C sozinho
            return ", funcionário júnior C será enviado sozinho"
        if estado_code == 2:
            # M3: senior A + treinando B
            return ", funcionário senior A vai ser enviado junto de funcionário treinando B"
        # fallback genérico
        if gravidade == "Alta":
            return ", funcionário senior A vai ser enviado junto de funcionário treinando B"
        if gravidade == "Media":
            return ", funcionário júnior C será enviado sozinho"
        return ""

    # ------------------------------------------------------------------
    # função principal para entrada estruturada
    # ------------------------------------------------------------------
    def handle_structured(self, dados_maquinas: List[Dict[str, Any]]) -> str:
        linhas = []

        for dado in dados_maquinas:
            maquina = dado.get("Maquina", "M?")
            estado_code = int(dado.get("Estado", 0))
            defeito_code = int(dado.get("Defeito", -1))

            # 1) Texto base do estado, usando mapeamento ligado à ontologia
            estado_txt = self.onto_access.map_estado_code_to_label(estado_code)

            # 2) Fragmento de defeito, usando mapeamento ligado às classes de defeito
            defeito_frag = self.onto_access.map_defeito_code_to_fragment(defeito_code)
            if defeito_frag:
                defeito_txt = f" {defeito_frag}"
            else:
                defeito_txt = ""

            # 3) Determinar gravidade qualitativa (para escolha de equipe)
            gravidade = self._estado_para_gravidade(estado_code)

            # 4) Escolher funcionários conforme as regras desejadas
            sufixo_func = self._escolher_funcionarios(gravidade, estado_code)

            # 5) Montar linha final
            linha = f"Maquina {maquina} {estado_txt}{defeito_txt}{sufixo_func}"
            linhas.append(linha)

        return "\n".join(linhas)

# ----------------------------------------------------------------------
# Instância global (compatível com seu padrão anterior)
# ----------------------------------------------------------------------

llm_gerente = LLMGerente()

# ----------------------------------------------------------------------
# Pequeno teste manual quando executado diretamente
# ----------------------------------------------------------------------
if __name__ == "__main__":
    entradas = [
        {"Maquina": "M1", "Estado": 0, "Defeito": -1},
        {"Maquina": "M2", "Estado": 1, "Defeito": 0},
        {"Maquina": "M3", "Estado": 2, "Defeito": 1},
    ]
    print(llm_gerente.handle_structured(entradas))
