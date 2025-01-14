import pandas as pd  # Import pandas for data handling

df = pd.read_csv('train.csv')  # Load the Titanic dataset

print(df)  # Display the whole dataset

print(df.head(10))  # Show the first 10 rows

print(df.tail(10))  # Show the last 10 rows

df.describe()  # Summary statistics for numeric columns

df.info()  # Dataset info: column types and non-null counts

df.shape  # Get the dataset's dimensions (rows, columns)

df.dtypes  # Display the data types of each column

print(df.isnull().sum())  # Count missing values in each column

mean_value = df['Age'].mean()  # Calculate the average age

median_value = df['Fare'].median()  # Calculate the median fare

mode_value = df['Embarked'].mode()  # Find the most common embarkation point

print("Mean Age =", mean_value)  # Print average age

print("Fare =", median_value)  # Print median fare

print("Mode Embarked", mode_value)  # Print most common embarkation point

df['Age'] = df['Age'].fillna(df['Age'].mean())  # Fill missing age values with the mean

