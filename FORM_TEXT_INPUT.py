# ---------------------------------------------------------------
# FORM_TEXT_INPUT.py
# Projeto-Mora — Integração de funcionários com a ontologia
# Entrada: lista de dicionários de funcionários (NÃO usa texto livre)
# ---------------------------------------------------------------

from typing import List, Dict, Any, Optional
from owlready2 import get_ontology

ONTOLOGY_PATH = "Project_Ontology.owx"  # ajuste se necessário


class FormTextInput:
    """
    Versão orientada a dados estruturados:
    - NÃO faz mais parsing de texto livre.
    - Recebe diretamente uma lista de dicionários de funcionários:
        [
          {'Nome': 'Beatrice', 'nivel': 'Senior'},
          {'Nome': 'Erika', 'nivel': 'Junior'},
          {'Nome': 'Eva', 'nivel': 'Treinando', 'Treinador': 'Beatrice'},
          ...
        ]
    - Cria indivíduos na ontologia Project_Ontology.
    - Retorna estrutura útil para o restante do sistema.
    """

    def __init__(self, ontology_path: str = ONTOLOGY_PATH):
        self.onto = get_ontology(ontology_path).load()

    # -----------------------------------------------------------
    # Entrada PRINCIPAL: lista de dicionários de funcionários
    # -----------------------------------------------------------
    def parse(self, funcionarios_data: List[Dict[str, Any]],
              salvar_arquivo: Optional[str] = None) -> Dict[str, Any]:
        """
        Entrada:
            funcionarios_data: lista de dicionários, ex.:
                [
                  {'Nome': 'Beatrice', 'nivel': 'Senior'},
                  {'Nome': 'Erika', 'nivel': 'Junior'},
                  {'Nome': 'Eva', 'nivel': 'Treinando', 'Treinador': 'Beatrice'},
                  ...
                ]

        Saída:
            {
              "individuos": { "Beatrice": <Beatrice>, ... },
              "resumo": {
                  "total": 10,
                  "seniors": 2,
                  "juniors": 1,
                  "treinandos": 3,
                  "contratados": 4
              },
              "treinamentos": {
                  "Beatrice": ["Eva"],
                  "Rosa": ["Kraus", "Rudolf"]
              }
            }
        """
        if not funcionarios_data:
            return {
                "individuos": {},
                "resumo": {"total": 0, "seniors": 0, "juniors": 0,
                           "treinandos": 0, "contratados": 0},
                "treinamentos": {}
            }

        individuos = self._criar_individuos_funcionarios(funcionarios_data)
        treinamentos = self._criar_relacoes_treinamento(funcionarios_data, individuos)

        resumo = self._resumir_niveis(funcionarios_data)

        if salvar_arquivo:
            self.onto.save(file=salvar_arquivo, format="rdfxml")

        return {
            "individuos": individuos,
            "resumo": resumo,
            "treinamentos": treinamentos,
        }

    # -----------------------------------------------------------
    # Criar indivíduos Funcionario/Senior/Junior/Treinando/Temporario
    # -----------------------------------------------------------
    def _criar_individuos_funcionarios(
        self, funcionarios_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        FuncionarioClass = self.onto.search_one(iri="*#Funcionario")
        SeniorClass = self.onto.search_one(iri="*#Senior")
        JuniorClass = self.onto.search_one(iri="*#Junior")
        TreinandoClass = self.onto.search_one(iri="*#Treinando")
        TemporarioClass = self.onto.search_one(iri="*#Temporario")  # para "Contratado"

        if FuncionarioClass is None:
            raise RuntimeError("Classe #Funcionario não encontrada na ontologia.")

        nivel_to_class = {
            "Senior": SeniorClass or FuncionarioClass,
            "Junior": JuniorClass or FuncionarioClass,
            "Treinando": TreinandoClass or FuncionarioClass,
            "Contratado": TemporarioClass or FuncionarioClass,
        }

        individuos: Dict[str, Any] = {}

        for f in funcionarios_data:
            nome = f.get("Nome")
            nivel = f.get("nivel")
            if not nome or not nivel:
                continue

            owl_cls = nivel_to_class.get(nivel, FuncionarioClass)
            safe_name = nome.replace(" ", "_")
            ind = owl_cls(safe_name)
            individuos[nome] = ind

        return individuos

    # -----------------------------------------------------------
    # Criar relações Treinado_por(Treinando, Senior)
    # -----------------------------------------------------------
    def _criar_relacoes_treinamento(
        self,
        funcionarios_data: List[Dict[str, Any]],
        individuos: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        Treinado_por = self.onto.search_one(iri="*#Treinado_por")
        mapa: Dict[str, List[str]] = {}

        if Treinado_por is None:
            return mapa

        for f in funcionarios_data:
            if f.get("nivel") == "Treinando" and "Treinador" in f:
                nome_treinando = f["Nome"]
                nome_treinador = f["Treinador"]
                if nome_treinando in individuos and nome_treinador in individuos:
                    treinando_ind = individuos[nome_treinando]
                    senior_ind = individuos[nome_treinador]
                    Treinado_por[treinando_ind].append(senior_ind)
                    mapa.setdefault(nome_treinador, []).append(nome_treinando)

        return mapa

    # -----------------------------------------------------------
    # Resumo numérico por nível
    # -----------------------------------------------------------
    def _resumir_niveis(self, funcionarios_data: List[Dict[str, Any]]) -> Dict[str, int]:
        seniors = sum(1 for f in funcionarios_data if f.get("nivel") == "Senior")
        juniors = sum(1 for f in funcionarios_data if f.get("nivel") == "Junior")
        treinandos = sum(1 for f in funcionarios_data if f.get("nivel") == "Treinando")
        contratados = sum(1 for f in funcionarios_data if f.get("nivel") == "Contratado")

        return {
            "total": len(funcionarios_data),
            "seniors": seniors,
            "juniors": juniors,
            "treinandos": treinandos,
            "contratados": contratados,
        }


# Instância global (para uso em LLM_G.py)
form_text_input = FormTextInput()


if __name__ == "__main__":
    # Teste rápido
    dados = [
        {'Nome': 'Beatrice', 'nivel': 'Senior'},
        {'Nome': 'Erika', 'nivel': 'Junior'},
        {'Nome': 'Eva', 'nivel': 'Treinando', 'Treinador': 'Beatrice'},
        {'Nome': 'George', 'nivel': 'Contratado'},
        {'Nome': 'Maria', 'nivel': 'Contratado'},
        {'Nome': 'Kraus', 'nivel': 'Treinando', 'Treinador': 'Rosa'},
        {'Nome': 'Rosa', 'nivel': 'Senior'},
        {'Nome': 'Rudolf', 'nivel': 'Treinando', 'Treinador': 'Rosa'},
        {'Nome': 'Jessica', 'nivel': 'Contratado'},
        {'Nome': 'Delta', 'nivel': 'Contratado'},
    ]

    resultado = form_text_input.parse(dados, salvar_arquivo="Project_Ontology_enriched.owl")
    print("Resumo:", resultado["resumo"])
    print("Treinamentos:", resultado["treinamentos"])
    print("Indivíduos criados:", list(resultado["individuos"].keys()))
