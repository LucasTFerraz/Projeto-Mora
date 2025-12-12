import sqlite3
from datetime import datetime


class DataBaseAcess:
    def __init__(self, db_path="project_mora.db"):
        """Inicializa conexão com banco SQLite."""
        self.db_path = db_path
        self._create_tables()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        """Cria as tabelas necessárias caso não existam."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                sensor_name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                predicted_value REAL NOT NULL,
                severity_level TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS machines (
                id TEXT PRIMARY KEY,
                description TEXT
            )
        """)

        conn.commit()
        conn.close()

    # ----------------------------------------------------------------------
    # INSERTS
    # ----------------------------------------------------------------------

    def insert_sensor_value(self, machine_id, sensor_name, value):
        """Insere leitura de sensor."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO sensors (machine_id, sensor_name, value, timestamp)
            VALUES (?, ?, ?, ?)
        """, (machine_id, sensor_name, value, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def insert_prediction(self, machine_id, model_name, predicted_value, severity_level):
        """Insere resultado da predição."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions (machine_id, model_name, predicted_value, severity_level, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (machine_id, model_name, predicted_value, severity_level, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    # ----------------------------------------------------------------------
    # QUERIES
    # ----------------------------------------------------------------------

    def get_latest_sensor_values(self, machine_id, limit=10):
        """Retorna últimas leituras de uma máquina."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT sensor_name, value, timestamp
            FROM sensors
            WHERE machine_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (machine_id, limit))

        rows = cursor.fetchall()
        conn.close()

        return rows

    def get_latest_prediction(self, machine_id):
        """Retorna última predição feita para uma máquina."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT model_name, predicted_value, severity_level, timestamp
            FROM predictions
            WHERE machine_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (machine_id,))

        row = cursor.fetchone()
        conn.close()

        return row

    def get_all_machines(self):
        """Retorna todas as máquinas cadastradas."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, description FROM machines
        """)

        rows = cursor.fetchall()
        conn.close()

        return rows

    def register_machine(self, machine_id, description=""):
        """Cadastra uma máquina se ainda não existir."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR IGNORE INTO machines (id, description)
            VALUES (?, ?)
        """, (machine_id, description))

        conn.commit()
        conn.close()
