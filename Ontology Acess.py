# Ontology_Acess.py
"""
Acesso à ontologia do Projeto-MORA.
Permite consultar classes, subclasses, indivíduos e propriedades da ontologia OWL.
"""
import types

from owlready2 import get_ontology, Thing, default_world

class OntologyAccess:
    def __init__(self, path: str = "Project_Ontology.owl"):
        self.path = path
        self.onto = get_ontology(path).load()

    # ----------------------------------------------------------------------
    # --- CONSULTAS BÁSICAS ---
    # ----------------------------------------------------------------------
    def add_Funcionario(self,func):
        with self.onto:
            funcionario = types.new_class("NewClassName", (Thing,))
    def list_classes(self):
        """Retorna todas as classes definidas na ontologia."""
        return [clazz for clazz in self.onto.classes()]

    def list_individuals(self):
        """Retorna todos os indivíduos existentes na ontologia."""
        return [ind for ind in self.onto.individuals()]

    def get_class(self, class_name: str):
        """Retorna uma classe pelo nome."""
        return getattr(self.onto, class_name, None)

    def get_individual(self, individual_name: str):
        """Retorna um indivíduo pelo nome."""
        return getattr(self.onto, individual_name, None)

    # ----------------------------------------------------------------------
    # --- RELAÇÕES ENTRE CLASSES ---
    # ----------------------------------------------------------------------

    def get_subclasses(self, class_name: str):
        """Retorna todas as subclasses de uma classe."""
        clazz = self.get_class(class_name)
        if clazz:
            return list(clazz.subclasses())
        return None

    def get_superclasses(self, class_name: str):
        """Retorna superclasses."""
        clazz = self.get_class(class_name)
        if clazz:
            return list(clazz.is_a)
        return None

    # ----------------------------------------------------------------------
    # --- INDIVÍDUOS, ATRIBUTOS E PROPRIEDADES ---
    # ----------------------------------------------------------------------

    def get_properties_of_class(self, class_name):
        """Lista todas as propriedades onde a classe aparece."""
        clazz = self.get_class(class_name)
        if clazz is None:
            return None
        return clazz.get_class_properties()

    def get_individual_properties(self, individual_name):
        """Lista propriedades e valores de um indivíduo."""
        ind = self.get_individual(individual_name)
        if ind is None:
            return None

        props = {}
        for prop in ind.get_properties():
            props[prop.name] = prop[ind]

        return props

    # ----------------------------------------------------------------------
    # --- CONSULTA DE ESTADOS, GRAVIDADES, DEFEITOS, FUNCIONÁRIOS ---
    # ----------------------------------------------------------------------

    def list_states(self):
        """Retorna todas as subclasses de Estado."""
        state_class = self.get_class("Estado")
        if state_class:
            return list(state_class.subclasses())
        return []

    def list_severities(self):
        """Retorna subclasses de Gravidade."""
        grav = self.get_class("Gravidade")
        if grav:
            return list(grav.subclasses())
        return []

    def list_defects(self):
        """Retorna subclasses de Defeito."""
        defect = self.get_class("Defeito")
        if defect:
            return list(defect.subclasses())
        return []

    def list_models(self):
        """Retorna subclasses de Modelo."""
        model = self.get_class("Modelo")
        if model:
            return list(model.subclasses())
        return []

    def list_machines(self):
        """Retorna subclasses de Maquina."""
        maq = self.get_class("Maquina")
        if maq:
            return list(maq.subclasses())
        return []

    def list_workers(self):
        """Retorna subclasses de Funcionario."""
        f = self.get_class("Funcionario")
        if f:
            return list(f.subclasses())
        return []

    # ----------------------------------------------------------------------
    # --- CONSULTAS SEMI-SPARQL ---
    # ----------------------------------------------------------------------

    def search(self, class_name=None, property_name=None, value=None):
        """
        Busca simples: por classe, propriedade ou ambos.
        """
        results = []

        # filtrar por classe
        if class_name:
            clazz = self.get_class(class_name)
            if clazz:
                for ind in clazz.instances():
                    results.append(ind)

        # filtrar por propriedade
        if property_name:
            prop = getattr(self.onto, property_name, None)
            if prop:
                for ind in prop.get_relations():
                    if value is None or value in prop[ind]:
                        results.append(ind)

        return list(set(results))

    # ----------------------------------------------------------------------
    # --- EXPORTA PARA A LLM (FORMATO LIMPO) ---
    # ----------------------------------------------------------------------

    def summarize(self):
        """
        Produz um dicionário com:
        - classes
        - indivíduos
        - estados
        - gravidades
        - defeitos
        - equipamentos
        - funcionários
        Usado para alimentar a LLM com contexto limpo e estruturado.
        """
        return {
            "Estados": [c.name for c in self.list_states()],
            "Gravidades": [c.name for c in self.list_severities()],
            "Defeitos": [c.name for c in self.list_defects()],
            "Modelos": [c.name for c in self.list_models()],
            "Maquinas": [c.name for c in self.list_machines()],
            "Funcionarios": [c.name for c in self.list_workers()],
        }


# Instância global da ontologia
ontology = OntologyAccess()
