import numpy as np


class PreProcessModel:
    """
    Responsável por:
    - receber leituras brutas do SensorInput
    - normalizar os dados
    - montar vetores adequados para cada modelo
    - retornar algo pronto para ModelSelector.predict()
    """

    def __init__(self, ontology=None):
        self.ontology = ontology

        # Normalização padrão (pode ser substituída por scaler treinado)
        self.default_ranges = {
            "temp": (0, 150),
            "vibration": (0, 12),
            "pressure": (0, 50),
            "oil_level": (0, 1)
        }

    # ----------------------------------------------------------------------
    # NORMALIZAÇÃO
    # ----------------------------------------------------------------------
    def normalize_value(self, sensor, value):
        if sensor not in self.default_ranges:
            # Se não temos range para o sensor, apenas escala 0–1 com fallback
            low, high = 0, 1
        else:
            low, high = self.default_ranges[sensor]

        norm = (value - low) / (high - low)
        return max(0.0, min(1.0, norm))

    # ----------------------------------------------------------------------
    # TRANSFORMA SENSORES EM VETOR PARA IA
    # ----------------------------------------------------------------------
    def to_vector(self, sensor_dict):
        """
        sensor_dict: {"temp": 80.2, "vibration": 0.34, ...}
        Retorna lista normalizada: [0.55, 0.20, ...]
        """

        vector = []

        for sensor, value in sensor_dict.items():
            normalized = self.normalize_value(sensor, value)
            vector.append(normalized)

        # CNN e RandomForest esperam vetor 1D
        # LSTM pode esperar shape (1, time_steps, features) – aqui usando 1 timestep
        return np.array(vector, dtype=np.float32)

    # ----------------------------------------------------------------------
    # PIPELINE COMPLETO CHAMADO NO main.py
    # ----------------------------------------------------------------------
    def preprocess(self, machine_data):
        """
        machine_data (vem de SensorInput.capture_machine_state):
        {
            "machine_id": "M1",
            "timestamp": "...",
            "sensors": { "temp": 80.2, "vibration": 0.33 }
        }

        Retorna estrutura completa:
        {
            "machine_id": "M1",
            "input_vector": numpy array [...],
            "raw_sensors": {...},
            "metadata": {...}
        }
        """

        machine_id = machine_data["machine_id"]
        sensors = machine_data["sensors"]

        vector = self.to_vector(sensors)

        enriched_metadata = {
            "timestamp": machine_data["timestamp"],
        }

        # Se Ontologia foi passada, enriquecer com conhecimento
        if self.ontology is not None:
            enriched_metadata.update(self.get_ontology_enrichment(machine_id))

        return {
            "machine_id": machine_id,
            "input_vector": vector,
            "raw_sensors": sensors,
            "metadata": enriched_metadata
        }

    # ----------------------------------------------------------------------
    # ENRIQUECIMENTO VIA ONTOLOGIA
    # ----------------------------------------------------------------------
    def get_ontology_enrichment(self, machine_id):
        """
        Obtém informações relevantes da ontologia (exemplos):
        - tipo da máquina
        - defeitos possíveis
        - sensores mais críticos
        - limiares de risco
        """
        try:
            return self.ontology.get_machine_context(machine_id)
        except Exception:
            return {}
