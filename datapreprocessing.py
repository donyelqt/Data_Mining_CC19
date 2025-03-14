import pandas as pd
from scipy import stats

df = pd.read_csv('assets/train.csv')

print('Dataset Overview:')
print(df.head(10))
print(df.tail(10))
print("\nDataset Summary:")
print(df.describe())
print(df.info())
print('\nMissing Values Count:')
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


df['Age'] = df['Age'].apply(lambda x: 80 if x > 80 else x)

z_scores = stats.zscore(df[['Age', 'Fare']].dropna())

df = df[(abs(z_scores) < 3).all(axis=1)]

df['Sex'] = df['Sex'].str.capitalize()
df['Embarked'] = df['Embarked'].str.upper()

