import pandas as pd


class CSVLoader:
    def __init__(self, path):
        self.path = path

    def to_text_list(self):
        df = pd.read_csv(self.path)

        # Está estático
        texts = [
            f"Data: {row['date']}. Temperatura média: {row['average_temperature']}°C. Umidade: {row['humidity']}%. Índice de ansiedade: {row['anxiety_index']}."
            for _, row in df.iterrows()
        ]
        return texts
