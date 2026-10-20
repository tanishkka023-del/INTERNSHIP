import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)

# 1. Generate Synthetic E-Commerce Spending Data
# Non-Members spend ~$150 on average, Members spend ~$175 on average
non_members = np.random.normal(loc=150, scale=35, size=250)
members = np.random.normal(loc=175, scale=40, size=250)

df = pd.DataFrame({
    'Spending': np.concatenate([non_members, members]),
    'Membership': ['Non-Member']*250 + ['Member']*250
})

# 2. Descriptive Statistics
summary_stats = df.groupby('Membership')['Spending'].agg(['count', 'mean', 'std', 'median'])
print("=== Descriptive Statistics ===")
print(summary_stats)
print("\n")

# 3. Hypothesis Testing: Independent Two-Sample T-Test (Welch's T-Test)
group_non_member = df[df['Membership'] == 'Non-Member']['Spending']
group_member = df[df['Membership'] == 'Member']['Spending']

t_stat, p_value = stats.ttest_ind(group_member, group_non_member, equal_var=False)

# Calculate 95% Confidence Intervals for Difference in Means
cm = sms.CompareMeans(sms.DescrStatsW(group_member), sms.DescrStatsW(group_non_member))
conf_int = cm.tconfint_diff(alpha=0.05, usevar='unequal')
mean_diff = group_member.mean() - group_non_member.mean()

print("=== Hypothesis Test Results ===")
print(f"Mean Difference (Member - Non-Member): ${mean_diff:.2f}")
print(f"T-Statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4e}")
print(f"95% Confidence Interval of Difference: ({conf_int[0]:.2f}, {conf_int[1]:.2f})")

# Decision Logic
alpha = 0.05
if p_value < alpha:
    print("\nDecision: Reject Null Hypothesis (H0).")
    print("Conclusion: Significant difference in spending between Members and Non-Members.")
else:
    print("\nDecision: Fail to Reject Null Hypothesis (H0).")

# 4. Visualizations
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Histogram / KDE Distribution
sns.histplot(data=df, x='Spending', hue='Membership', kde=True, ax=axes[0], palette='Set2', element="step")
axes[0].set_title('Distribution of Customer Spending by Membership', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Annual Spending ($)')
axes[0].set_ylabel('Frequency')

# Plot 2: Boxplot with Mean Indicators
sns.boxplot(data=df, x='Membership', y='Spending', ax=axes[1], palette='Set2', showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
axes[1].set_title('Spending Distribution & Mean Comparison', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Membership Status')
axes[1].set_ylabel('Annual Spending ($)')

plt.tight_layout()
plt.savefig('hypothesis_testing_plots.png', dpi=300)
plt.show()
