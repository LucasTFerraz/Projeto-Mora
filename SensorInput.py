import random
from datetime import datetime
from DataBaseAcess import DataBaseAcess


class SensorInput:
    def __init__(self, db: DataBaseAcess):
        self.db = db

        # Sensores conhecidos por máquina (exemplo)
        self.machine_sensors = {
            "M1": ["temp", "vibration", "pressure"],
            "M2": ["temp", "vibration"],
            "M3": ["temp", "vibration", "oil_level"]
        }

    # ----------------------------------------------------------------------
    # LEITURA DE SENSORES (ATUAL OU SIMULADA)
    # ----------------------------------------------------------------------
    def read_sensor(self, machine_id, sensor_name):
        """
        Simula ou lê um sensor real.
        Aqui está simulando valores randômicos.
        Pode ser substituído por leitura real de hardware/OPC-UA/MQTT/etc.
        """

        # Exemplos de ranges realistas:
        ranges = {
            "temp": (20, 120),
            "vibration": (0.0, 10.0),
            "pressure": (1.0, 30.0),
            "oil_level": (0.0, 1.0)
        }

        low, high = ranges.get(sensor_name, (0, 1))
        value = round(random.uniform(low, high), 3)
        return value

    def read_all_sensors(self, machine_id):
        """
        Lê todos os sensores da máquina e retorna dict:
        {
            "temp": 80.1,
            "vibration": 0.34,
            ...
        }
        """

        if machine_id not in self.machine_sensors:
            raise ValueError(f"Sensores não definidos para a máquina {machine_id}")

        readings = {}

        for sensor in self.machine_sensors[machine_id]:
            value = self.read_sensor(machine_id, sensor)
            readings[sensor] = value

            # Salva no banco
            self.db.insert_sensor_value(machine_id, sensor, value)

        return readings

    # ----------------------------------------------------------------------
    # FLUXO COMPLETO: LEITURA + ARMAZENAMENTO + RETORNO
    # ----------------------------------------------------------------------
    def capture_machine_state(self, machine_id):
        """
        Função usada pelo main.py:
        - lê sensores
        - salva no banco
        - retorna dict formatado para o PreProcess_UseModel

        Exemplo de retorno:
        {
            "machine_id": "M1",
            "timestamp": "...",
            "sensors": {
                "temp": 85.2,
                "vibration": 1.23
            }
        }
        """
        sensor_values = self.read_all_sensors(machine_id)

        return {
            "machine_id": machine_id,
            "timestamp": datetime.now().isoformat(),
            "sensors": sensor_values
        }
