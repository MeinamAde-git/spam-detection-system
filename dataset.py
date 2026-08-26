import pandas as pd

def create_dataset():

    df = pd.read_csv('spam.csv', encoding='latin-1')

    df = df[['v1', 'v2']]

    df.columns = ['label', 'email']

    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    df = df.dropna().reset_index(drop=True)

    print(f"✅ Dataset loaded successfully!")
    print(f"   Total emails : {len(df)}")
    print(f"   Spam emails  : {df['label'].sum()}")
    print(f"   Ham emails   : {(df['label'] == 0).sum()}")

    return df2


if __name__ == "__main__":
    df = create_dataset()
    print("\nSample Data:")
    print(df.head())