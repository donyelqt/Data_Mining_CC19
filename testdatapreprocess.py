import pandas as pd
from scipy import stats

# ===============================
# 1. Load and Inspect the Dataset
# ===============================
df = pd.read_csv('assets/train.csv')  # Load the Titanic dataset

# Initial Dataset Overview
print("Dataset Overview:")
print(df.head(10))  # Display the first 10 rows
print(df.tail(10))  # Display the last 10 rows
print("\nDataset Summary:")
print(df.describe())  # Summary statistics for numeric columns
print(df.info())  # Dataset info: column types and non-null counts

# Missing values overview
print("\nMissing Values Count:")
print(df.isnull().sum())  # Count missing values in each column

# ===============================
# 2. Fill Missing Values
# ===============================
df['Age'] = df['Age'].fillna(df['Age'].mean())  # Fill missing Age values with the mean
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Fill missing Embarked values with the mode

# ===============================
# 3. Removing Duplicate Rows
# ===============================
df.drop_duplicates(inplace=True)
print(f"\nNumber of duplicate rows remaining: {df.duplicated().sum()}")  # Should be 0

# ===============================
# 4. Handling Outliers
# ===============================
# Cap values in the Age column
df['Age'] = df['Age'].apply(lambda x: 80 if x > 80 else x)

# Use Z-score to detect and remove outliers for numerical columns
z_scores = stats.zscore(df[['Age', 'Fare']].dropna())  # Drop NaN before calculation
df = df[(abs(z_scores) < 3).all(axis=1)]

print("\nAfter outlier removal:")
print(df[['Age', 'Fare']].describe())  # Review numerical data post-cleaning

# ===============================
# 5. Standardizing Categorical Data
# ===============================
# Capitalize values in the "Sex" column
df['Sex'] = df['Sex'].str.capitalize()

# Convert "Embarked" column values to uppercase
df['Embarked'] = df['Embarked'].str.upper()

# Check the unique values in categorical columns
print("\nUnique values in categorical columns:")
print("Sex:", df['Sex'].unique())
print("Embarked:", df['Embarked'].unique())

# ===============================
# 6. Dataset Final Review and Summary
# ===============================
# Missing values check
print("\nMissing values count after cleaning:")
print(df.isnull().sum())

# Summary statistics for numeric columns
print("\nCleaned Dataset Summary:")
print(df.describe())

# ===============================
# 7. Save the Cleaned Dataset
# ===============================
df.to_csv('assets/cleaned_titanic.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_titanic.csv'")
