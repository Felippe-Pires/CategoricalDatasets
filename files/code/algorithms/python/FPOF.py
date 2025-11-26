import os
import pandas as pd

from utils.loading import load_from_csv_dataframe
from algorithms.FPOF.fpof import FPOF

# Exemplo de uso:
if __name__ == "__main__":
    # Substitua os dados de exemplo pelos seus próprios dados
    data = pd.DataFrame({
        'A=x': [1, 0, 1, 0],
        'B=y': [0, 1, 1, 0],
        'C=z': [1, 1, 0, 1]
    })

    file_dir = os.path.abspath(os.path.dirname(__file__))

    DATA_PATH = os.path.join(file_dir, 'database', 'CBRW_paper_example.csv')
    EXCLUDE_COLS = ['Cheat?']
    X = load_from_csv_dataframe(DATA_PATH, header=True, exclude_cols=EXCLUDE_COLS)

    result = FPOF(X, minSupport=0.1, mlen=3)
    print(result)
