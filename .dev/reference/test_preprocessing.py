import pandas as pd
import numpy as np

# Column definitions for Adult Census Income dataset
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

def load_and_preprocess_adult(file_path="adult.data"):
    # Load dataset
    df = pd.read_csv(
        file_path,
        names=COLUMNS,
        sep=r',\s*',
        engine='python',
        na_values='?'
    )
    
    print(f"Initial shape: {df.shape}")
    
    # Drop missing values
    df = df.dropna().reset_index(drop=True)
    print(f"Shape after dropping missing values: {df.shape}")
    
    # Drop fnlwgt (administrative sampling weight, not an interpretable personal feature)
    # Also drop redundant education-num since education text is much more interpretable
    df = df.drop(columns=['fnlwgt', 'education-num'])
    
    # Clean target column (strip extra dots or spaces)
    df['income'] = df['income'].astype(str).str.rstrip('.')
    print(f"Class distribution:\n{df['income'].value_counts(normalize=True)}")
    
    # Continuous variable discretization into human-interpretable categories
    # 1. Age
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 25, 45, 65, 100],
        labels=['<=25 (Young)', '26-45 (Middle-aged)', '46-65 (Senior-Adult)', '>65 (Elderly)'],
        right=True
    ).astype(str)
    
    # 2. Hours per week
    df['hours_group'] = pd.cut(
        df['hours-per-week'],
        bins=[0, 34, 40, 55, 100],
        labels=['<35 (Part-time)', '35-40 (Full-time)', '41-55 (Overtime)', '>55 (Workaholic)'],
        right=True
    ).astype(str)
    
    # 3. Capital Gain (Heavily skewed, >90% zero)
    df['capital_gain_group'] = pd.cut(
        df['capital-gain'],
        bins=[-1, 0, 7000, 100000],
        labels=['None', 'Moderate (<7k)', 'High (>7k)'],
        right=True
    ).astype(str)
    
    # 4. Capital Loss (Heavily skewed, >95% zero)
    df['capital_loss_group'] = pd.cut(
        df['capital-loss'],
        bins=[-1, 0, 1800, 100000],
        labels=['None', 'Low-Loss (<=1.8k)', 'High-Loss (>1.8k)'],
        right=True
    ).astype(str)
    
    # Drop raw continuous columns
    df = df.drop(columns=['age', 'hours-per-week', 'capital-gain', 'capital-loss'])
    
    print(f"Processed feature columns: {df.columns.tolist()}")
    return df

if __name__ == '__main__':
    df = load_and_preprocess_adult()
    print(df.head())
