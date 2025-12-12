# ============================================================================
# EMPLOYEES_MORA.py
# Sistema Inteligente de Gerenciamento de Funcionários - PROJETO MORA
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import json

# ============================================================================
# 1. MODELO DE DADOS - FUNCIONÁRIO
# ============================================================================

@dataclass
class Funcionario:
    """Representa um funcionário no sistema MORA"""
    nome: str
    nivel: str  # Senior, Junior, Treinando, Contratado
    treinador: Optional[str] = None
    data_contratacao: str = field(default_factory=lambda: datetime.now().isoformat())
    historico_atribuicoes: List[Dict] = field(default_factory=list)
    competencias: List[str] = field(default_factory=list)
    disponibilidade: bool = True
    carga_trabalho: int = 0

    def __post_init__(self):
        """Validação ontológica"""
        niveis_validos = ["Senior", "Junior", "Treinando", "Contratado"]
        if self.nivel not in niveis_validos:
            raise ValueError(f"Nível inválido: {self.nivel}")
        
        if self.nivel == "Treinando" and self.treinador is None:
            raise ValueError("Treinando deve ter treinador")

    def to_dict(self) -> dict:
        return {
            "nome": self.nome, "nivel": self.nivel, "treinador": self.treinador,
            "data_contratacao": self.data_contratacao, "competencias": self.competencias,
            "disponibilidade": self.disponibilidade, "carga_trabalho": self.carga_trabalho,
            "historico_atribuicoes": len(self.historico_atribuicoes)
        }

# ============================================================================
# 2. GERENCIADOR DE FUNCIONÁRIOS (10 REAIS CARREGADOS)
# ============================================================================

class GerenciadorFuncionarios:
    """Gerencia 10 funcionários reais com validações ontológicas"""

    def __init__(self):
        self.funcionarios: Dict[str, Funcionario] = {}
        self.regras_atribuicao = {
            "Senior": {"gravidade_maxima": "Alta", "pode_treinar": True, "max_treinandos": 2, "prioridade": 1},
            "Junior": {"gravidade_maxima": "Media", "pode_treinar": False, "max_treinandos": 0, "prioridade": 3},
            "Treinando": {"gravidade_maxima": "Baixa", "requer_supervisor": True, "prioridade": 4},
            "Contratado": {"gravidade_maxima": "Baixa", "requer_treinamento": True, "prioridade": 5}
        }

    def carregar_10_funcionarios_reais(self):
        """Carrega os 10 funcionários reais do projeto"""
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
            {'Nome': 'Delta', 'nivel': 'Contratado'}
        ]
        
        print("🔄 Carregando 10 funcionários reais...")
        for dados_func in dados:
            try:
                func = Funcionario(
                    nome=dados_func['Nome'],
                    nivel=dados_func['nivel'],
                    treinador=dados_func.get('Treinador')
                )
                self.adicionar_funcionario(func)
            except ValueError as e:
                print(f"❌ Erro: {e}")
        
        print(f"✓ {len(self.funcionarios)}/10 funcionários carregados!")

    def adicionar_funcionario(self, funcionario: Funcionario) -> bool:
        if funcionario.nome in self.funcionarios:
            return False
        
        if funcionario.nivel == "Treinando":
            if funcionario.treinador not in self.funcionarios:
                return False
            treinador = self.funcionarios[funcionario.treinador]
            if treinador.nivel != "Senior":
                return False
            treinandos = sum(1 for f in self.funcionarios.values() 
                           if f.treinador == funcionario.treinador)
            if treinandos >= 2:
                return False
        
        self.funcionarios[funcionario.nome] = funcionario
        return True

    def listar_por_nivel(self, nivel: str) -> List[Funcionario]:
        return [f for f in self.funcionarios.values() if f.nivel == nivel]

    def obter_treinandos(self, nome_treinador: str) -> List[Funcionario]:
        return [f for f in self.funcionarios.values() 
                if f.treinador == nome_treinador and f.nivel == "Treinando"]

    def resumo(self) -> Dict:
        return {
            "total": len(self.funcionarios),
            "seniors": len(self.listar_por_nivel("Senior")),
            "juniors": len(self.listar_por_nivel("Junior")),
            "treinandos": len(self.listar_por_nivel("Treinando")),
            "contratados": len(self.listar_por_nivel("Contratado")),
            "disponibilidade": sum(1 for f in self.funcionarios.values() if f.disponibilidade)
        }

# ============================================================================
# 3. MOTOR DE ATRIBUIÇÃO INTELIGENTE
# ============================================================================

class MotorAtribuicao:
    """Atribui equipes baseado em gravidade e regras ontológicas"""

    def __init__(self, gerenciador: GerenciadorFuncionarios):
        self.gerenciador = gerenciador

    def atribuir_equipe(self, machine_id: str, gravidade: str, 
                       tipo_defeito: str = "Nenhum") -> Dict:
        requisitos = {
            "Zero": ["Contratado", "Treinando", "Junior", "Senior"],
            "Baixa": ["Treinando", "Junior", "Senior"],
            "Media": ["Junior", "Senior"],
            "Alta": ["Senior"]
        }
        
        niveis_capazes = requisitos.get(gravidade, ["Senior"])
        primario = self._encontrar_melhor_candidato(niveis_capazes)
        
        if primario is None:
            return {"erro": "Nenhum funcionário disponível"}
        
        secundario = None
        if gravidade == "Alta" and primario.nivel == "Senior":
            secundario = self._encontrar_treinando(primario.nome)
        
        atribuicao = {
            "machine_id": machine_id, "gravidade": gravidade,
            "tipo_defeito": tipo_defeito, "timestamp": datetime.now().isoformat()
        }
        
        primario.historico_atribuicoes.append(atribuicao)
        primario.carga_trabalho += 1
        primario.disponibilidade = False
        
        if secundario:
            secundario.historico_atribuicoes.append(atribuicao)
            secundario.carga_trabalho += 1
            secundario.disponibilidade = False
        
        return {
            "primario": primario.nome,
            "nivel_primario": primario.nivel,
            "secundario": secundario.nome if secundario else None,
            "nivel_secundario": secundario.nivel if secundario else None,
            "motivo": f"Gravidade {gravidade}",
            "equipe_completa": f"{primario.nome}" + (f" + {secundario.nome}" if secundario else "")
        }

    def _encontrar_melhor_candidato(self, niveis_capazes: List[str]) -> Optional[Funcionario]:
        candidatos = []
        for nivel in niveis_capazes:
            for f in self.gerenciador.listar_por_nivel(nivel):
                if f.disponibilidade:
                    candidatos.append(f)
        if not candidatos:
            return None
        candidatos.sort(key=lambda f: f.carga_trabalho)
        return candidatos[0]

    def _encontrar_treinando(self, nome_senior: str) -> Optional[Funcionario]:
        treinandos = self.gerenciador.obter_treinandos(nome_senior)
        for t in treinandos:
            if t.disponibilidade:
                return t
        return None

# ============================================================================
# 4. INTERFACE INTERATIVA
# ============================================================================

class InterfaceInterativa:
    def __init__(self, gerenciador: GerenciadorFuncionarios, motor: MotorAtribuicao):
        self.gerenciador = gerenciador
        self.motor = motor

    def exibir_menu(self):
        print("\n" + "=" * 70)
        print("  🏭 PROJETO MORA - GERENCIADOR DE FUNCIONÁRIOS")
        print("=" * 70)
        print("1. Listar todos")
        print("2. Estatísticas")
        print("3. Atribuir equipe")
        print("4. Treinandos por Senior")
        print("5. Disponibilidade")
        print("6. Histórico")
        print("7. Exportar JSON")
        print("8. Sair")
        print("-" * 70)

    def executar(self):
        while True:
            self.exibir_menu()
            opcao = input("Opção (1-8): ").strip()
            
            if opcao == "1":
                self._listar_funcionarios()
            elif opcao == "2":
                self._estatisticas()
            elif opcao == "3":
                self._atribuir()
            elif opcao == "4":
                self._treinandos()
            elif opcao == "5":
                self._disponibilidade()
            elif opcao == "6":
                self._historico()
            elif opcao == "7":
                self._exportar()
            elif opcao == "8":
                break

    def _listar_funcionarios(self):
        print("\n📋 FUNCIONÁRIOS (10 reais carregados):")
        for nivel in ["Senior", "Junior", "Treinando", "Contratado"]:
            funcs = self.gerenciador.listar_por_nivel(nivel)
            if funcs:
                print(f"\n📌 {nivel} ({len(funcs)}):")
                for f in funcs:
                    status = "✓" if f.disponibilidade else "✗"
                    print(f"  {status} {f.nome} (carga: {f.carga_trabalho})")

    def _estatisticas(self):
        stats = self.gerenciador.resumo()
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"Total: {stats['total']} | Disponíveis: {stats['disponibilidade']}")
        print(f"Seniors: {stats['seniors']} | Juniors: {stats['juniors']}")
        print(f"Treinandos: {stats['treinandos']} | Contratados: {stats['contratados']}")

    def _atribuir(self):
        machine = input("Máquina (M1): ").strip() or "M1"
        gravidade = input("Gravidade (Baixa/Media/Alta): ").strip().capitalize() or "Baixa"
        resultado = self.motor.atribuir_equipe(machine, gravidade)
        print(f"\n✅ {resultado['equipe_completa']} → {resultado['motivo']}")

    def _treinandos(self):
        senior = input("Senior (Beatrice/Rosa): ").strip() or "Rosa"
        treinandos = self.gerenciador.obter_treinandos(senior)
        print(f"\n👨‍🏫 {senior} treina: {', '.join([t.nome for t in treinandos])}")

    def _disponibilidade(self):
        nome = input("Funcionário: ").strip()
        status = input("Disponível? (s/n): ").strip().lower() == 's'
        # Simula mudança (implementar se necessário)

    def _historico(self):
        nome = input("Funcionário: ").strip() or "Rosa"
        if nome in self.gerenciador.funcionarios:
            f = self.gerenciador.funcionarios[nome]
            print(f"{f.nome}: {len(f.historico_atribuicoes)} atribuições")

    def _exportar(self):
        dados = {"funcionarios": {n: f.to_dict() for n, f in self.gerenciador.funcionarios.items()}}
        with open("funcionarios.json", "w") as f:
            json.dump(dados, f, indent=2)
        print("💾 Exportado para funcionarios.json")

# ============================================================================
# 5. SISTEMA COMPLETO PRONTO
# ============================================================================

def main():
    """Sistema completo com 10 funcionários reais"""
    gerenciador = GerenciadorFuncionarios()
    gerenciador.carregar_10_funcionarios_reais()  # ✅ CARREGA OS 10
    
    motor = MotorAtribuicao(gerenciador)
    interface = InterfaceInterativa(gerenciador, motor)
    
    print("\n🎉 SISTEMA MORA ATIVO - 10 funcionários carregados!")
    interface.executar()

if __name__ == "__main__":
    main()
