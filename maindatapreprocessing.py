import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('assets/train.csv')

print('Dataset Overview:')
print(df.head(10))
print(df.tail(10))
print('\nDataset Summary:')
print(df.describe())
print(df.info())

print('\nMissing Values Count:')
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df.drop_duplicates(inplace=True)
print(f'\nNumber of duplicate rows remaining: {df.duplicated().sum()}')

df['Age'] = df['Age'].apply(lambda x: 80 if x > 80 else x)

z_scores = stats.zscore(df[['Age', 'Fare',]].dropna())
df = df[(abs(z_scores) < 3).all(axis=1)]

print('\nAfter outlier removal:')
print(df[['Age', 'Fare']].describe())

df['Sex'] = df['Sex'].str.capitalize()
df['Embarked'] = df['Embarked'].str.upper()

print('\nUnique values in catagorical columns:')
print('Sex', df['Sex'].unique())
print('Embarked', df['Embarked'].unique())

print('\nMissing values count after cleaning:')
print(df.isnull().sum())

print('\nCleaned Dataset Summary:')
print(df.describe())

df.to_csv('assets/new_cleaned_titanic.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_titanic.csv'")

# visualize outliers with boxplots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Age'], color='skyblue')
plt.title('Boxplot of Age')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['Fare'], color='lightgreen')
plt.title('Boxplot of Fare')
plt.tight_layout()
plt.show()
