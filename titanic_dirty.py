# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Load the dataset
df = pd.read_csv('assets/titanic_dirty.csv')

# Initial Dataset Overview
print("Initial Dataset Info:")
print(df.info())
print("\nInitial Summary Statistics:")
print(df.describe())

# Removing Duplicates
print("\nChecking for duplicates...")
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
removed_duplicates = initial_rows - df.shape[0]
print(f"Number of duplicate rows removed: {removed_duplicates}")
print(f"Number of duplicates remaining: {df.duplicated().sum()}")

# Handling Missing Values
df['Age'] = df['Age'].fillna(df['Age'].mean())  # Fill missing 'Age' values with mean
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Fill missing 'Embarked' with mode

# Handling Outliers
# Method 1: Capping Age values greater than 80
df['Age'] = df['Age'].apply(lambda x: 80 if x > 80 else x)

# Method 2: Capping Fare values greater than 150
df['Fare'] = df['Fare'].apply(lambda x: 160 if x > 160 else x)

# Handling Negative Outliers: Setting negative values to 1
df['Age'] = df['Age'].apply(lambda x: 1 if x < 0 else x)
df['Fare'] = df['Fare'].apply(lambda x: 1 if x < 0 else x)

# Visualizing Outliers using Seaborn Boxplots
plt.figure(figsize=(10, 5))

# Boxplot for Age (after capping and handling negative outliers)
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Age'], color='lightblue')
plt.title('Boxplot of Age (Capped at 80, Negative Outliers to 1)')
plt.ylabel('Age')

# Boxplot for Fare (after capping and handling negative outliers)
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Fare'], color='lightgreen')
plt.title('Boxplot of Fare (Capped at 160, Negative Outliers to 1)')
plt.ylabel('Fare')

plt.tight_layout()
plt.show()

# Method 3: Z-Score Method (Optional for further outlier handling)
z_scores = stats.zscore(df[['Age', 'Fare']])
df = df[(abs(z_scores) < 3).all(axis=1)]

print("\nSummary Statistics After Handling Outliers:")
print(df[['Age', 'Fare']].describe())

# Standardizing Categorical Data
df['Sex'] = df['Sex'].str.capitalize()  # Ensure "male" becomes "Male"
df['Embarked'] = df['Embarked'].str.upper()  # Ensure "C" or "S" is uppercase

# Checking Unique Values in Categorical Columns
print("\nUnique Values in 'Sex':", df['Sex'].unique())
print("Unique Values in 'Embarked':", df['Embarked'].unique())

# Final Dataset Overview
print("\nFinal Dataset Info:")
print(df.info())
print("\nFinal Summary Statistics:")
print(df.describe())

# Save the cleaned dataset
df.to_csv('cleaned_titanic.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_titanic.csv'.")


