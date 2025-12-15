# LLM_G.py
# Gerente inteligente do Projeto MORA (versão orientada a funcionários)
# Entrada: lista de dicionários de funcionários
# Saída: texto com interpretação + integração na ontologia

from typing import List, Dict, Any
from FORM_TEXT_INPUT import form_text_input
from Ontology_Acess import OntologyAccess
from DataBaseAcess import DataBaseAcess


class LLMGerente:
    """
    Esta versão do LLMGerente NÃO recebe texto livre.
    Em vez disso, recebe diretamente uma lista de dicionários de funcionários:
        [
          {'Nome': 'Beatrice', 'nivel': 'Senior'},
          {'Nome': 'Erika', 'nivel': 'Junior'},
          {'Nome': 'Eva', 'nivel': 'Treinando', 'Treinador': 'Beatrice'},
          ...
        ]

    Ela:
    1) Cria indivíduos correspondentes na ontologia (via FORM_TEXT_INPUT.funcionarios_para_ontologia)
    2) Valida hierarquia: Seniors, Juniors, Treinandos, Contratados
    3) Gera um resumo textual da estrutura de treinamento
    """

    def __init__(self):
        self.onto_access = OntologyAccess()
        self.db = DataBaseAcess()

    # ------------------------------------------------------------
    # Entrada: lista de dicionários de funcionários
    # ------------------------------------------------------------
    def handle(self, funcionarios_data: List[Dict[str, Any]]) -> str:
        """
        Entrada típica:
        [
          {'Nome': 'Beatrice', 'nivel': 'Senior'},
          {'Nome': 'Erika', 'nivel': 'Junior'},
          {'Nome': 'Eva', 'nivel': 'Treinando', 'Treinador': 'Beatrice'},
          ...
        ]

        Saída: string com resumo da hierarquia e do que foi criado na ontologia.
        """

        if not funcionarios_data:
            return "Nenhum funcionário recebido."

        # 1) Criar indivíduos na ontologia via FORM_TEXT_INPUT
        individuos = form_text_input.funcionarios_para_ontologia(
            funcionarios_data,
            salvar_arquivo=None  # ou "Project_Ontology_enriched.owl"
        )

        # 2) Construir mapa hierárquico em memória
        seniors = []
        juniors = []
        treinandos = []
        contratados = []

        mapa_treinandos: Dict[str, List[str]] = {}

        for f in funcionarios_data:
            nome = f.get("Nome")
            nivel = f.get("nivel")
            treinador = f.get("Treinador")

            if nivel == "Senior":
                seniors.append(nome)
            elif nivel == "Junior":
                juniors.append(nome)
            elif nivel == "Treinando":
                treinandos.append(nome)
                if treinador:
                    mapa_treinandos.setdefault(treinador, []).append(nome)
            elif nivel == "Contratado":
                contratados.append(nome)

        # 3) Montar texto de saída
        linhas: List[str] = []

        linhas.append("Funcionários recebidos e inseridos na ontologia:")
        for nome in individuos.keys():
            linhas.append(f"- {nome}")

        linhas.append("")
        linhas.append("Resumo por nível:")
        linhas.append(f"- Seniors: {', '.join(seniors) if seniors else 'nenhum'}")
        linhas.append(f"- Juniors: {', '.join(juniors) if juniors else 'nenhum'}")
        linhas.append(f"- Treinandos: {', '.join(treinandos) if treinandos else 'nenhum'}")
        linhas.append(f"- Contratados: {', '.join(contratados) if contratados else 'nenhum'}")

        linhas.append("")
        linhas.append("Estrutura de treinamento (Senior → Treinandos):")
        if mapa_treinandos:
            for senior, ts in mapa_treinandos.items():
                linhas.append(f"- {senior} treina: {', '.join(ts)}")
        else:
            linhas.append("- Nenhuma relação de treinamento encontrada.")

        return "\n".join(linhas)

    # ------------------------------------------------------------
    # Compatibilidade com API antiga (processar)
    # ------------------------------------------------------------
    def processar(self, mensagem, contexto_extra=None):
        """
        Mantido apenas para compatibilidade com main.py antigo.
        Aqui, IGNORAMOS 'mensagem' e usamos apenas 'contexto_extra'
        como lista de dicionários de funcionários.
        """
        if isinstance(contexto_extra, list):
            return self.handle(contexto_extra)
        return "Contexto extra deve ser uma lista de funcionários."


# Instância global
llm_gerente = LLMGerente()


if __name__ == "__main__":
    # Exemplo de uso direto
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

    print(llm_gerente.handle(dados))
