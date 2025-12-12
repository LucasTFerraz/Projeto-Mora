# ---------------------------------------------------------------
# FORM_TEXT_INPUT.py - VERSÃO INTEGRADA COM FUNCIONÁRIOS
# Projeto-Mora — Processamento inicial de texto do operador
# + Gerenciamento de funcionários reais
# ---------------------------------------------------------------

import re
from typing import List, Dict, Any, Optional


# ============================================================================
# CLASSE PRINCIPAL ATUALIZADA
# ============================================================================

class FormTextInput:
    """
    Processador híbrido: Texto livre + Lista de funcionários reais
    Retorna dados estruturados para LLM_Gerente
    """

    def __init__(self, funcionarios_data: Optional[List[Dict[str, Any]]] = None):
        """
        Inicializa com mapeamentos + opcionalmente lista de funcionários

        Args:
            funcionarios_data: Lista dos 10 funcionários reais
        """
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
            "etc": "ETC",
            "quente": "Temperatura",
            "fumaça": "Falha_Crítica",
            "fumaça": "Falha_Crítica"
        }

        # possíveis causas
        self.map_causas = {
            "eixo x": "Eixo_X",
            "eixo y": "Eixo_Y",
            "eixo z": "Eixo_Z",
            "s_i": "S_i",
            "s_j": "S_j",
            "s_k": "S_k",
            "rolamento": "Rolamento",
            "motor": "Motor",
            "correia": "Correia"
        }

        # ✅ CARREGA FUNCIONÁRIOS REAIS
        self.funcionarios = self._carregar_funcionarios(funcionarios_data)
        print(f"✓ FormTextInput inicializado com {len(self.funcionarios)} funcionários")

    def _carregar_funcionarios(self, dados: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Carrega e indexa os 10 funcionários reais por nome"""
        if not dados:
            return {}

        funcionarios = {}
        for i, dados_func in enumerate(dados):
            nome = dados_func.get('Nome', f'Funcionario_{i}')
            funcionarios[nome] = {
                'nome': nome,
                'nivel': dados_func.get('nivel', 'Desconhecido'),
                'treinador': dados_func.get('Treinador', None)
            }
        return funcionarios

    # -----------------------------------------------------------
    # Função principal - AGORA RECEBE FUNCIONÁRIOS!
    # -----------------------------------------------------------
    def parse(self, text_input: str, funcionarios_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        ✅ VERSÃO ATUALIZADA: Recebe texto + funcionários → dados estruturados + contexto

        Args:
            text_input: Texto livre do operador
            funcionarios_data: Lista dos 10 funcionários (opcional, atualiza self.funcionarios)

        Returns:
            {
                "maquina": "M1",
                "caracteristica": "Barulho",
                "causa": "Eixo_Y",
                "texto_original": "...",
                "funcionarios": { "Beatrice": {...}, "Rosa": {...} },  ← NOVO!
                "total_funcionarios": 10,                              ← NOVO!
                "seniors_disponiveis": 2                               ← NOVO!
            }
        """
        # Atualizar lista de funcionários se fornecida
        if funcionarios_data:
            self.funcionarios = self._carregar_funcionarios(funcionarios_data)

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
        if causa is None:
            causa = "Desconhecida"

        # ----------------------------
        # 4) CONTEXTO DE FUNCIONÁRIOS - NOVO!
        # ----------------------------
        seniors = [f for f in self.funcionarios.values() if f['nivel'] == 'Senior']
        treinandos = [f for f in self.funcionarios.values() if f['nivel'] == 'Treinando']

        # Resumo para LLMGerente
        contexto_funcionarios = {
            "total": len(self.funcionarios),
            "seniors": len(seniors),
            "juniors": len([f for f in self.funcionarios.values() if f['nivel'] == 'Junior']),
            "treinandos": len(treinandos),
            "contratados": len([f for f in self.funcionarios.values() if f['nivel'] == 'Contratado']),
            "seniors_disponiveis": len(seniors),  # Simula todos disponíveis
            "exemplos": {
                "senior": seniors[0]['nome'] if seniors else None,
                "treinando": treinandos[0]['nome'] if treinandos else None
            }
        }

        # ----------------------------
        # 5) RESULTADO ESTRUTURADO COMPLETO
        # ----------------------------
        resultado = {
            "maquina": maquina,
            "caracteristica": caracteristica,
            "causa": causa,
            "texto_original": text_input,
            "funcionarios": self.funcionarios,  # ← Lista completa dos 10
            "contexto": contexto_funcionarios  # ← Resumo otimizado
        }

        return resultado

    # -----------------------------------------------------------
    # MÉTODOS AUXILIARES PARA LLMGerente
    # -----------------------------------------------------------
    def get_seniors(self) -> List[str]:
        """Retorna nomes dos Seniors"""
        return [f['nome'] for f in self.funcionarios.values() if f['nivel'] == 'Senior']

    def get_treinandos(self, senior_nome: str) -> List[str]:
        """Retorna treinandos de um Senior específico"""
        return [f['nome'] for f in self.funcionarios.values()
                if f['nivel'] == 'Treinando' and f['treinador'] == senior_nome]

    def resumo_funcionarios(self) -> Dict[str, int]:
        """Resumo rápido para debugging"""
        niveis = {}
        for f in self.funcionarios.values():
            nivel = f['nivel']
            niveis[nivel] = niveis.get(nivel, 0) + 1
        return niveis


# ============================================================================
# INSTÂNCIA GLOBAL (Compatibilidade com LLM_G.py original)
# ============================================================================

# Dados dos 10 funcionários reais
DADOS_FUNCIONARIOS = [
    {'Nome': 'Beatrice', 'nivel': 'Senior'},
    {'Nome': 'Erika', 'nivel': 'Junior'},
    {'Nome': 'Eva', 'nivel': 'Treinando', 'Treinador': 'Beatrice'},
    {'Nome': 'George', 'nivel': 'Contratado'},
    {'Nome': 'Maria', 'nivel': 'Contratado'},
    {'Nome': 'Kraus', 'nivel': 'Treinando', 'Treinador': 'Rosa'},
    {'Nome': 'Rosa', 'nivel': 'Senior'},
    {'Nome': 'Rudolf', 'nivel': 'Treinando', 'Treinador': 'Rosa'},
    {'Nome': 'Jessica', 'nivel': 'Contratado'},
    {'Nome': 'Delta', 'nivel': 'Contratado'}
]

# Instância global com os 10 funcionários carregados
form_text_input = FormTextInput(DADOS_FUNCIONARIOS)

# ============================================================================
# TESTE RÁPIDO
# ============================================================================
if __name__ == "__main__":
    print("🚀 FORM_TEXT_INPUT INTEGRADO - Teste com 10 funcionários")
    print("=" * 60)

    # Teste 1: Texto normal
    resultado1 = form_text_input.parse("M1 está barulhento, eixo y")
    print("\n📨 TESTE 1: 'M1 está barulhento, eixo y'")
    print(f"   Máquina: {resultado1['maquina']}")
    print(f"   Característica: {resultado1['caracteristica']}")
    print(f"   Causa: {resultado1['causa']}")
    print(f"   Total funcionários: {resultado1['contexto']['total']}")
    print(f"   Seniors disponíveis: {resultado1['contexto']['seniors_disponiveis']}")

    # Teste 2: Atualizar lista de funcionários
    print("\n📨 TESTE 2: Atualizando lista de funcionários...")
    nova_lista = DADOS_FUNCIONARIOS[:5]  # Primeiros 5 apenas
    resultado2 = form_text_input.parse("M2 vibrando", nova_lista)
    print(f"   Novo total: {resultado2['contexto']['total']} funcionários")

    # Teste 3: Resumo
    print("\n📊 RESUMO DOS FUNCIONÁRIOS:")
    resumo = form_text_input.resumo_funcionarios()
    for nivel, count in resumo.items():
        print(f"   {nivel}: {count}")

    print("\n✅ FormTextInput pronto para LLM_Gerente!")
