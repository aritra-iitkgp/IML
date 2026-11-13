
import numpy as np
import pandas as pd
import torch

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

AGE_BINS = [0, 25, 45, 65, 120]
AGE_LABELS = ['Young (<=25)', 'Middle-aged (26-45)', 'Senior-Adult (46-65)', 'Senior (>65)']

HOURS_BINS = [0, 34, 40, 55, 120]
HOURS_LABELS = ['Part-time (<35)', 'Full-time (35-40)', 'Overtime (41-55)', 'Extreme (>55)']

GAIN_BINS = [-1, 0, 7000, 1_000_000]
GAIN_LABELS = ['None', 'Moderate (<7k)', 'High (>7k)']

LOSS_BINS = [-1, 0, 1800, 1_000_000]
LOSS_LABELS = ['None', 'Low-Loss (<=1.8k)', 'High-Loss (>1.8k)']


def load_adult(file_path="data/adult.data", verbose=True):
    """Read the Adult file, treat '?' as missing, drop incomplete rows, encode target."""
    df = pd.read_csv(file_path, names=COLUMNS, sep=r',\s*', engine='python',
                     na_values=['?', ' ?'])

    # Strip whitespace from the text columns (some values are padded).
    # IMPORTANT: preserve NaN values so dropna() can remove incomplete rows.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_after = len(df)

    # The test file's target sometimes has a trailing '.', so normalise it here too.
    df['income'] = df['income'].astype(str).str.rstrip('.')
    y = (df['income'] == '>50K').astype(int).values
    X = df.drop(columns=['income'])

    if verbose:
        print(f"Rows before dropna : {n_before}")
        print(f"Rows after  dropna : {n_after}")
        pos = y.mean()
        print(f"Class balance      : <=50K {1 - pos:.2%} | >50K {pos:.2%}")
    return X, y


def discretise(df):
    """Turn the raw frame into an all-categorical frame of readable literals.

    Design choices:
      - drop fnlwgt         : survey sampling weight, not a personal attribute.
      - drop education-num  : redundant with the readable `education` text.
      - bin age / hours / capital-gain / capital-loss into domain categories.
      - collapse native-country to United-States vs Other (long tail).
    """
    df = df.copy()
    df = df.drop(columns=['fnlwgt', 'education-num'])

    df['age_group'] = pd.cut(df['age'].astype(float), bins=AGE_BINS,
                             labels=AGE_LABELS, right=True).astype(str)
    df['hours_group'] = pd.cut(df['hours-per-week'].astype(float), bins=HOURS_BINS,
                               labels=HOURS_LABELS, right=True).astype(str)
    df['capital_gain_group'] = pd.cut(df['capital-gain'].astype(float), bins=GAIN_BINS,
                                      labels=GAIN_LABELS, right=True).astype(str)
    df['capital_loss_group'] = pd.cut(df['capital-loss'].astype(float), bins=LOSS_BINS,
                                      labels=LOSS_LABELS, right=True).astype(str)

    df = df.drop(columns=['age', 'hours-per-week', 'capital-gain', 'capital-loss'])
    df['native_country_group'] = np.where(df['native-country'] == 'United-States',
                                          'United-States', 'Other')
    df = df.drop(columns=['native-country'])
    return df


class LiteralEncoder:
    """Encode a categorical frame as a boolean coverage matrix in PyTorch.

    A literal is a (column, value) pair, e.g. ('education', 'Bachelors').
    M[i, j] == True means literal i is satisfied by example j.
    A rule is the element-wise AND of the rows of M it selects.
    """

    def fit(self, X_df):
        self.columns_ = list(X_df.columns)
        self.literals_ = [(c, v) for c in self.columns_ for v in pd.unique(X_df[c])]
        self.col_index_ = {c: i for i, c in enumerate(self.columns_)}
        self.literal_col_ = torch.tensor(
            [self.col_index_[c] for c, _ in self.literals_], dtype=torch.long)
        return self

    def encode(self, X_df):
        rows = [X_df[c].values == v for c, v in self.literals_]
        return torch.from_numpy(np.stack(rows))          # bool (L, N)

    def rule_mask(self, M, literal_idx):
        if len(literal_idx) == 0:
            return torch.ones(M.shape[1], dtype=torch.bool)
        return M[list(literal_idx)].all(dim=0)

    def text(self, literal_idx):
        return " AND ".join(f"[{c} = '{v}']" for c, v in
                            (self.literals_[i] for i in literal_idx))


def literal_index_map(enc):
    """Map (column, value) -> literal row index, for rebuilding coverage."""
    return {lit: i for i, lit in enumerate(enc.literals_)}