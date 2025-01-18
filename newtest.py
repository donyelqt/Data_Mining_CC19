import pandas as pd
from scipy import stats

# ===============================
# 1. Load the Dataset
# ===============================
df = pd.read_csv('assets/train.csv')  # Load the Titanic dataset

print("\nInitial Dataset Overview:")
print(f"Shape of dataset: {df.shape}")  # Rows and columns count
print("Preview of dataset:")
print(df.head())

# ===============================
# 2. Removing Duplicates
# ===============================
print("\nChecking for duplicate rows...")
# Identify duplicate rows
duplicates = df[df.duplicated()]
print(f"Number of duplicate rows: {duplicates.shape[0]}")

# Remove duplicate rows
df.drop_duplicates(inplace=True)
print(f"Dataset shape after removing duplicates: {df.shape}")

# ===============================
# 3. Detecting Outliers
# ===============================
# a. Boxplot method (visual)
print("\nDetecting outliers using a visual approach (Boxplot)...")
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot for Age and Fare
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Age'], color='skyblue')
plt.title('Boxplot of Age')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['Fare'], color='lightgreen')
plt.title('Boxplot of Fare')
plt.tight_layout()
plt.show()

# b. Z-Score Method (programmatic)
print("\nDetecting outliers using the Z-score method...")
z_scores = stats.zscore(df[['Age', 'Fare']].dropna())  # Calculate Z-scores for Age and Fare
threshold = 3  # Common Z-score threshold
outliers = (abs(z_scores) > threshold).any(axis=1)  # Identify rows with Z-scores > 3
print(f"Number of outliers detected: {outliers.sum()}")

# ===============================
# 4. Handling Outliers
# ===============================
print("\nHandling outliers...")

# a. Removing rows with outliers
df_no_outliers = df[~outliers]
print(f"Dataset shape after removing outliers: {df_no_outliers.shape}")

# b. Capping Outliers (Alternative to removal)
df_capped = df.copy()
df_capped['Age'] = df_capped['Age'].apply(lambda x: 80 if x > 80 else x)  # Cap Age at 80
df_capped['Fare'] = df_capped['Fare'].apply(lambda x: 500 if x > 500 else x)  # Cap Fare at 500
print(f"Shape of dataset after capping outliers: {df_capped.shape}")

# ===============================
# 5. Final Review and Summary
# ===============================
print("\nFinal Dataset Summary:")
print("Dataset after removing duplicates and handling outliers (by removal):")
print(df_no_outliers.describe())

print("\nDataset after capping outliers:")
print(df_capped[['Age', 'Fare']].describe())

# Save cleaned datasets
df_no_outliers.to_csv('assets/titanic_no_outliers.csv', index=False)
df_capped.to_csv('assets/titanic_capped_outliers.csv', index=False)
print("\nCleaned datasets saved as 'titanic_no_outliers.csv' and 'titanic_capped_outliers.csv'.")
