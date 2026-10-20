import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# 1. DATA ACQUISITION
# ---------------------------------------------------------
# Using the standard Titanic dataset as a publicly available benchmark
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("--- Initial Data Summary ---")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------------
# 2. DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------
# A. Remove Duplicates
df.drop_duplicates(inplace=True)

# B. Handle Missing Values
# - Age: Impute missing values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# - Embarked: Impute missing categorical values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# - Cabin: Drop column due to excessive missing data (>70%)
df.drop(columns=['Cabin'], inplace=True)

# C. Correct Data Types
df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')
df['Sex'] = df['Sex'].astype('category')

print("\n--- Data Cleaning Complete ---")
print(df.isnull().sum())

# ---------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS & VISUALIZATIONS
# ---------------------------------------------------------

# Visualization 1: Missing Values Heatmap (Post-cleaning verification)
plt.figure(figsize=(8, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Visualization 1: Post-Cleaning Null Value Map")
plt.tight_layout()
plt.savefig("vis1_missing_values.png")
plt.show()

# Visualization 2: Distribution & Boxplot of Numerical Feature (Age)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['Age'], kde=True, color='skyblue')
plt.title("Age Distribution")

plt.subplot(1, 2, 2)
sns.boxplot(y=df['Age'], color='lightgreen')
plt.title("Age Outlier Analysis")
plt.suptitle("Visualization 2: Numerical Feature Distribution & Outlier Detection")
plt.tight_layout()
plt.savefig("vis2_distribution_outliers.png")
plt.show()

# Visualization 3: Correlation Matrix Heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Visualization 3: Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("vis3_correlation_heatmap.png")
plt.show()

# Summary Statistics
print("\n--- Summary Statistics ---")
print(df.describe())
