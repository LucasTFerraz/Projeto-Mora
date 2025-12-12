# LLM_G.py
# Gerente inteligente do Projeto MORA
# Combina: Ontologia + Modelos ML + Regras + Entrada Estruturada

from FORM_TEXT_INPUT import form_text_input
from OntologyAcess import OntologyAccess
from model_select import model_select
from DataBaseAcess import DataBaseAccess


class LLMGerente:

    def __init__(self):
        self.onto = OntologyAccess()
        self.db = DataBaseAccess()

    # ------------------------------------------------------------
    # Interpretação da entrada: passa pelo parser
    # ------------------------------------------------------------
    def parse_input(self, text: str):
        return form_text_input.parse(text)

    # ------------------------------------------------------------
    # Determinar o estado da máquina:
    #   - Ontologia
    #   - Banco de dados
    #   - Modelos ML (CNN/LSTM/RF)
    #   - Regras padrão (fallback)
    #
    # Isso permite funcionar mesmo antes de você enviar os indivíduos.
    # ------------------------------------------------------------
    def determine_machine_status(self, machine: str):
        # 1) Tenta ontologia
        state = self.onto.get_machine_state(machine)
        if state:
            return state

        # 2) Tenta banco
        db_state = self.db.get_machine_state(machine)
        if db_state:
            return db_state

        # 3) Tenta modelos (exemplo usando random forest)
        try:
            pred = model_select.predict("rf", [[0, 0, 0, 0]])
            if pred:
                return pred[0]
        except:
            pass

        # 4) Fallback: regras demonstrativas
        if machine.lower() == "m1":
            return "Normal"

        if machine.lower() == "m2":
            return "Distante_de_Quebrar"

        if machine.lower() == "m3":
            return "Perto_de_Quebrar"

        return "Desconhecido"

    # ------------------------------------------------------------
    # Determinar gravidade
    # ------------------------------------------------------------
    def determine_gravity(self, machine: str):
        # Busca pela ontologia primeiro
        gravity = self.onto.get_machine_gravity(machine)
        if gravity:
            return gravity

        # Regras padrão
        status = self.determine_machine_status(machine)

        if status == "Normal":
            return "Baixa"
        if status == "Distante_de_Quebrar":
            return "Média"
        if status == "Perto_de_Quebrar":
            return "Alta"

        return "Desconhecida"

    # ------------------------------------------------------------
    # Definir funcionários a enviar
    # ------------------------------------------------------------
    def assign_employees(self, machine: str):
        status = self.determine_machine_status(machine)
        gravity = self.determine_gravity(machine)

        # Regras descritas por você:

        # Caso simples → júnior sozinho
        if status in ["Normal", "Distante_de_Quebrar"] and gravity in ["Baixa", "Média"]:
            return ["C (júnior)"]

        # Caso intermediário → sênior + treinando
        if status in ["Perto_de_Quebrar"] or gravity == "Alta":
            return ["A (sênior)", "B (treinando)"]

        # Caso desconhecido
        return ["A (sênior)"]

    # ------------------------------------------------------------
    # Mensagem final
    # ------------------------------------------------------------
    def build_output(self, machine: str):
        status = self.determine_machine_status(machine)
        gravity = self.determine_gravity(machine)
        employees = self.assign_employees(machine)

        # Tradução simples
        if status == "Distante_de_Quebrar":
            status_desc = "ainda está longe de quebrar"
        elif status == "Perto_de_Quebrar":
            status_desc = "está se aproximando de quebrar"
        elif status == "Normal":
            status_desc = "está normal"
        else:
            status_desc = f"está em estado {status}"

        # Montagem
        if len(employees) == 1:
            emp_desc = f"funcionário {employees[0]} será enviado sozinho"
        else:
            emp_desc = f"funcionário {employees[0]} vai ser enviado junto de {employees[1]}"

        return f"Máquina {machine} {status_desc}, {emp_desc}."

    # ------------------------------------------------------------
    # Função principal
    # ------------------------------------------------------------
    def handle(self, text: str):
        parsed = self.parse_input(text)

        outputs = []
        for machine in parsed["machines"]:
            outputs.append(self.build_output(machine))

        if not outputs:
            return "Nenhuma máquina identificada no texto."

        return "\n".join(outputs)


# Instância global
llm_gerente = LLMGerente()
