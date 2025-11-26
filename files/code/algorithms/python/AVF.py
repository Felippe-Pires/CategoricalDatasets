# https://github.com/prk327/AtConP/blob/master/R/Attribute_Value_Frequency.r

import pandas as pd
import string
import random

def discretize_numerical_values(df, column, bins, labels):
    """
    Discretiza os valores de uma coluna em um DataFrame.

    Args:
    - df (pandas.DataFrame): DataFrame contendo os dados.
    - column (str): Nome da coluna a ser discretizada.
    - bins (int ou sequência de escalares): Número de bins ou os limites dos bins.
    - labels (sequência ou bool): Rótulos para os bins discretizados.

    Returns:
    - pandas.DataFrame: DataFrame com a coluna discretizada adicionada.
    """
    df[column] = pd.cut(df[column], bins=bins, labels=labels)
    return df

def generate_random_label(k=2):
    """Função para gerar um rótulo aleatório de duas letras."""
    return ''.join(random.choices(string.ascii_lowercase, k=k))

def AVF(df, bins=10, size_label=2):
    m = len(df.columns)
    lines = len(df)

    # Discretização dos valores contínuos
    for c in df.columns:
        labels = [generate_random_label(size_label) for _ in range(0, bins)]
        while len(set(labels)) < bins:
            labels.append(generate_random_label(size_label))
        labels = list(set(labels))
        if str(df[c].dtypes) == 'float' or str(df[c].dtypes) == 'float64':
            df = discretize_numerical_values(df, c, bins=bins,
                                             labels=labels)

    # Cálculo das frequências
    freq = {}
    for c in df.columns:
        freq[c] = df[c].value_counts().to_dict()

    # Somatório de frequência das linhas
    colunas = df.columns
    scores = []
    for row in df.itertuples(index=True, name='Pandas'):
        somatorio = 0
        for c in colunas:
            somatorio += freq[c][getattr(row, c)]/lines
        scores.append(somatorio/m)
    df['Score'] = scores

    return df



# Exemplo de uso:
if __name__ == "__main__":
    # Exemplo de uso
    #data = {'A': ['foo', 'bar', 'baz', 'foo', 'bar', 'baz'], 'B': [0.1, 0.4, 0.6, 0.1, 0.2, 0.3]}
    #df = pd.DataFrame(data)

    #resultado = AVF(df)
    #print(resultado)

    pass

