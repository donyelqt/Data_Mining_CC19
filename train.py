import pandas as pd

# Load the Titanic dataset
df = pd.read_csv('assets/train.csv')

print(df.head(5))
df.describe()

print(df.tail(5))
df.describe()
df.info()
df.shape
df.dtypes
print(df.isnull().sum())

# Calculate overall statistics
mean = df.mean(numeric_only=True)
median = df.median(numeric_only=True)
mode = df.mode().iloc[0]

print("Mean:\n", mean)
print("\nMedian:\n", median)
print("\nMode (first row):\n", mode)

# Calculate statistics for 'Age' column
age_mean = df['Age'].mean()
age_median = df['Age'].median()
age_mode = df['Age'].mode()[0]

print(f"\nAge - Mean: {age_mean}, Median: {age_median}, Mode: {age_mode}")

# Analyze the 'Embarked' column
embarked_unique = df['Embarked'].unique()
embarked_counts = df['Embarked'].value_counts()
embarked_missing = df['Embarked'].isnull().sum()

print(f"Embarked - Unique Values: {embarked_unique}")
print(f"Embarked - Value Counts:\n{embarked_counts}")
print(f"Embarked - Missing Values: {embarked_missing}")

print(df['Embarked'])